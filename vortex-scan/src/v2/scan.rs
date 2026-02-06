// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::Range;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use futures::Stream;
use futures::future::BoxFuture;
use vortex_array::ArrayFuture;
use vortex_array::ArrayRef;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::stream::ArrayStream;
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_layout::v2::reader::ReaderRef;
use vortex_layout::v2::reader::ReaderStreamRef;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::Selection;

pub struct ScanBuilder2 {
    reader: ReaderRef,
    projection: Expression,
    filter: Option<Expression>,
    limit: Option<u64>,
    row_range: Range<u64>,
    row_selection: Selection, // NOTE: applies to the selected row range.
    session: VortexSession,
}

impl ScanBuilder2 {
    pub fn new(reader: ReaderRef, session: VortexSession) -> Self {
        let row_range = 0..reader.row_count();
        Self {
            reader,
            projection: root(),
            filter: None,
            limit: None,
            row_range,
            row_selection: Selection::All,
            session,
        }
    }

    pub fn with_filter(mut self, filter: Expression) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_some_filter(mut self, filter: Option<Expression>) -> Self {
        self.filter = filter;
        self
    }

    pub fn with_projection(mut self, projection: Expression) -> Self {
        self.projection = projection;
        self
    }

    pub fn with_limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn with_row_range(mut self, row_range: Range<u64>) -> Self {
        self.row_range = row_range;
        self
    }

    /// Sets the row selection to use the given selection (relative to the row range).
    pub fn with_row_selection(mut self, row_selection: Selection) -> Self {
        self.row_selection = row_selection;
        self
    }

    /// Sets the row selection to include only the given row indices (relative to the row range).
    pub fn with_row_indices(mut self, row_indices: Buffer<u64>) -> Self {
        self.row_selection = Selection::IncludeByIndex(row_indices);
        self
    }

    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream> {
        let projection = self.projection.optimize_recursive(self.reader.dtype())?;
        let filter = self
            .filter
            .map(|f| f.optimize_recursive(self.reader.dtype()))
            .transpose()?;

        let dtype = projection.return_dtype(self.reader.dtype())?;

        // Apply expressions to the reader tree.
        let projection_reader = self.reader.apply(&projection)?;
        let filter_reader = filter.as_ref().map(|f| self.reader.apply(f)).transpose()?;

        tracing::info!(
            "Executing scan with:\nProjection:\n{}\nFilter:\n{}",
            projection_reader.display_tree(),
            filter_reader
                .as_ref()
                .map_or("None".to_string(), |f| f.display_tree().to_string())
        );

        // Execute both readers over the row range to produce streams.
        // TODO(ngates): we could partition this?
        let row_offset = self.row_range.start;
        let filter_stream = filter_reader
            .map(|r| r.execute(self.row_range.clone()))
            .transpose()?;
        let projection_stream = projection_reader.execute(self.row_range)?;

        Ok(Scan {
            dtype,
            filter_stream,
            projection_stream,
            pending: None,
            limit: self.limit,
            rows_produced: 0,
            row_selection: self.row_selection,
            row_offset,
            filter_buffer: None,
            projection_buffer: None,
        })
    }
}

struct Scan {
    dtype: DType,
    filter_stream: Option<ReaderStreamRef>,
    projection_stream: ReaderStreamRef,
    pending: Option<BoxFuture<'static, VortexResult<ArrayRef>>>,
    limit: Option<u64>,
    rows_produced: u64,
    row_selection: Selection,
    row_offset: u64,
    filter_buffer: Option<ArrayFuture>,
    projection_buffer: Option<ArrayFuture>,
}

impl Scan {
    /// Get the next projection chunk, consuming from the buffer first.
    fn next_projection_chunk(&mut self) -> Option<VortexResult<ArrayFuture>> {
        if let Some(buffered) = self.projection_buffer.take() {
            return Some(Ok(buffered));
        }
        self.projection_stream.next_chunk()
    }

    /// Get the next filter chunk, consuming from the buffer first.
    fn next_filter_chunk(
        stream: &mut ReaderStreamRef,
        buffer: &mut Option<ArrayFuture>,
    ) -> Option<VortexResult<ArrayFuture>> {
        if let Some(buffered) = buffer.take() {
            return Some(Ok(buffered));
        }
        stream.next_chunk()
    }

    /// Collect filter chunks covering exactly `n` rows.
    /// Returns a vec of ArrayFutures whose total len == n.
    fn collect_filter_chunks(&mut self, n: usize) -> Option<VortexResult<Vec<ArrayFuture>>> {
        let filter_stream = self.filter_stream.as_mut()?;
        let mut chunks = Vec::new();
        let mut remaining = n;

        while remaining > 0 {
            let chunk = match Self::next_filter_chunk(filter_stream, &mut self.filter_buffer) {
                Some(Ok(f)) => f,
                Some(Err(e)) => return Some(Err(e)),
                None => {
                    return Some(Err(vortex_error::vortex_err!(
                        "Filter stream exhausted before covering {} rows",
                        n
                    )));
                }
            };

            if chunk.len() <= remaining {
                remaining -= chunk.len();
                chunks.push(chunk);
            } else {
                // Buffer the remainder.
                let used = chunk.slice(0..remaining);
                self.filter_buffer = Some(chunk.slice(remaining..chunk.len()));
                remaining = 0;
                chunks.push(used);
            }
        }

        Some(Ok(chunks))
    }

    /// Skip `n` rows in the filter stream, consuming from the buffer first.
    fn skip_filter(&mut self, n: usize) {
        let mut remaining = n;
        if let Some(buf) = self.filter_buffer.take() {
            if remaining < buf.len() {
                self.filter_buffer = Some(buf.slice(remaining..buf.len()));
                return;
            }
            remaining -= buf.len();
        }
        if remaining > 0
            && let Some(filter_stream) = &mut self.filter_stream
        {
            filter_stream.skip(remaining);
        }
    }
}

impl ArrayStream for Scan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl Stream for Scan {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            // Poll pending future if we have one.
            if let Some(fut) = this.pending.as_mut() {
                match fut.as_mut().poll(cx) {
                    Poll::Ready(result) => {
                        this.pending = None;
                        return match result {
                            Ok(array) => {
                                this.rows_produced += array.len() as u64;
                                Poll::Ready(Some(Ok(array)))
                            }
                            Err(e) => Poll::Ready(Some(Err(e))),
                        };
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }

            // Check limit.
            if this.limit.is_some_and(|limit| this.rows_produced >= limit) {
                return Poll::Ready(None);
            }

            // Get the next projection chunk.
            let proj_future = match this.next_projection_chunk() {
                Some(Ok(f)) => f,
                Some(Err(e)) => return Poll::Ready(Some(Err(e))),
                None => return Poll::Ready(None),
            };

            let mut chunk_len = proj_future.len();

            // Apply limit: if remaining < chunk_len, slice the projection future.
            if let Some(limit) = this.limit {
                let remaining = usize::try_from(limit - this.rows_produced).unwrap_or(usize::MAX);
                if remaining < chunk_len {
                    // Buffer the remainder and use only what we need.
                    this.projection_buffer = Some(proj_future.slice(remaining..chunk_len));
                    chunk_len = remaining;
                }
            }

            let proj_future = if chunk_len < proj_future.len() {
                proj_future.slice(0..chunk_len)
            } else {
                proj_future
            };

            if chunk_len == 0 {
                return Poll::Ready(None);
            }

            // Compute the selection mask for this chunk's row range.
            let chunk_row_range = this.row_offset..this.row_offset + chunk_len as u64;
            let selection_mask = this.row_selection.row_mask(&chunk_row_range).mask().clone();

            // If all rows are excluded by selection, skip this chunk entirely (no I/O).
            if selection_mask.all_false() {
                this.skip_filter(chunk_len);
                this.row_offset += chunk_len as u64;
                continue;
            }

            this.row_offset += chunk_len as u64;

            // Build the pending future: await projection, apply filter + selection.
            if this.filter_stream.is_some() {
                let filter_chunks = match this.collect_filter_chunks(chunk_len) {
                    Some(Ok(chunks)) => chunks,
                    Some(Err(e)) => return Poll::Ready(Some(Err(e))),
                    None => {
                        return Poll::Ready(Some(Err(vortex_error::vortex_err!(
                            "Filter stream missing"
                        ))));
                    }
                };

                this.pending = Some(Box::pin(async move {
                    // Await filter chunks and combine into a single mask.
                    let mut filter_masks: Vec<Mask> = Vec::with_capacity(filter_chunks.len());
                    for filter_chunk in filter_chunks {
                        let filter_array = filter_chunk.await?;
                        filter_masks.push(filter_array.try_to_mask_fill_null_false()?);
                    }

                    let filter_mask = if filter_masks.len() == 1 {
                        filter_masks
                            .into_iter()
                            .next()
                            .unwrap_or_else(|| Mask::new_true(0))
                    } else {
                        Mask::concat(filter_masks.iter())?
                    };

                    // AND with selection mask.
                    let mask = if selection_mask.all_true() {
                        filter_mask
                    } else {
                        (&filter_mask).bitand(&selection_mask)
                    };

                    // Await projection.
                    let array = proj_future.await?;

                    // Filter the projection array.
                    if mask.all_true() {
                        Ok(array)
                    } else {
                        array.filter(mask)
                    }
                }));
            } else if selection_mask.all_true() {
                // No filter, no selection masking — just await projection.
                this.pending = Some(Box::pin(proj_future));
            } else {
                // No filter, but selection mask needs to be applied.
                this.pending = Some(Box::pin(async move {
                    let array = proj_future.await?;
                    array.filter(selection_mask)
                }));
            }

            // Loop back to poll the newly created future.
        }
    }
}

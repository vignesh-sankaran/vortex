// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::Range;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use futures::Stream;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::stream::ArrayStream;
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_layout::v2::reader::ReaderRef;
use vortex_layout::v2::reader::ReaderStreamRef;
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
        let filter_reader = filter.as_ref().map(|f| self.reader.apply(f)).transpose()?;
        let projection_reader = self.reader.apply(&projection)?;

        // Execute both readers over the row range to produce streams.
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

            // Determine the next chunk size from the projection stream.
            let proj_chunk_len = this.projection_stream.next_chunk_len();

            // If a filter stream exists, synchronize chunk sizes.
            let chunk_len = if let Some(filter_stream) = &this.filter_stream {
                match (proj_chunk_len, filter_stream.next_chunk_len()) {
                    (Some(p), Some(f)) => Some(p.min(f)),
                    _ => None,
                }
            } else {
                proj_chunk_len
            };

            let Some(mut chunk_len) = chunk_len else {
                return Poll::Ready(None);
            };

            // Limit the chunk size to avoid exceeding the row limit.
            if let Some(limit) = this.limit {
                let remaining = usize::try_from(limit - this.rows_produced).unwrap_or(usize::MAX);
                chunk_len = chunk_len.min(remaining);
                if chunk_len == 0 {
                    return Poll::Ready(None);
                }
            }

            // Compute the selection mask for this chunk's row range.
            let chunk_row_range = this.row_offset..this.row_offset + chunk_len as u64;
            let selection_mask = this.row_selection.row_mask(&chunk_row_range).mask().clone();

            // If all rows are excluded by selection, skip this chunk entirely.
            if selection_mask.all_false() {
                if let Some(filter_stream) = &mut this.filter_stream {
                    filter_stream.skip(chunk_len);
                }
                this.projection_stream.skip(chunk_len);
                this.row_offset += chunk_len as u64;
                continue;
            }

            // Build the mask chain: evaluate filter (if present), then AND with selection.
            let mask = if let Some(filter_stream) = &mut this.filter_stream {
                let all_true = MaskFuture::new_true(chunk_len);
                let filter_fut = match filter_stream.next_chunk(all_true) {
                    Ok(fut) => fut,
                    Err(e) => return Poll::Ready(Some(Err(e))),
                };
                MaskFuture::new(chunk_len, async move {
                    let filter_result = filter_fut.await?;
                    let filter_mask = filter_result.try_to_mask_fill_null_false()?;
                    if selection_mask.all_true() {
                        Ok(filter_mask)
                    } else {
                        Ok((&filter_mask).bitand(&selection_mask))
                    }
                })
            } else if selection_mask.all_true() {
                MaskFuture::new_true(chunk_len)
            } else {
                MaskFuture::ready(selection_mask)
            };

            this.row_offset += chunk_len as u64;

            // Request the next projection chunk with the computed mask.
            this.pending = Some(match this.projection_stream.next_chunk(mask) {
                Ok(fut) => fut,
                Err(e) => return Poll::Ready(Some(Err(e))),
            });

            // Loop back to poll the newly created future.
        }
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::future::try_join_all;
use moka::future::FutureExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::ChunkedArray;
use vortex_array::expr::Expression;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStream;
use crate::v2::reader::ReaderStreamRef;

/// A reader over a sequence of chunk readers with the same dtype.
pub struct ChunkedReader {
    row_count: u64,
    dtype: DType,
    chunks: Vec<ReaderRef>,
}

impl ChunkedReader {
    pub fn new(row_count: u64, dtype: DType, chunks: Vec<ReaderRef>) -> Self {
        Self {
            row_count,
            dtype,
            chunks,
        }
    }
}

impl Reader for ChunkedReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn apply(&self, expression: &Expression) -> VortexResult<ReaderRef> {
        let new_chunks: Vec<ReaderRef> = self
            .chunks
            .iter()
            .map(|chunk| chunk.apply(expression))
            .collect::<VortexResult<_>>()?;
        let new_dtype = expression.return_dtype(&self.dtype)?;
        Ok(Arc::new(Self {
            row_count: self.row_count,
            dtype: new_dtype,
            chunks: new_chunks,
        }))
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        // Collect overlapping chunk readers (not streams — lazy construction).
        let mut remaining_start = row_range.start;
        let mut remaining_end = row_range.end;
        let mut pending_chunks = VecDeque::new();

        for chunk in &self.chunks {
            let chunk_row_count = chunk.row_count();

            if remaining_start >= chunk_row_count {
                remaining_start -= chunk_row_count;
                remaining_end -= chunk_row_count;
                continue;
            }

            let start_in_chunk = remaining_start;
            let end_in_chunk = remaining_end.min(chunk_row_count);

            pending_chunks.push_back(PendingChunk {
                reader: chunk.clone(),
                row_range: start_in_chunk..end_in_chunk,
            });

            remaining_start = 0;
            if remaining_end <= chunk_row_count {
                break;
            } else {
                remaining_end -= chunk_row_count;
            }
        }

        Ok(Box::new(ChunkedReaderStream {
            dtype: self.dtype.clone(),
            pending_chunks,
            active_stream: None,
        }))
    }
}

struct PendingChunk {
    reader: ReaderRef,
    row_range: Range<u64>,
}

/// A stream that lazily constructs child streams as it advances through chunks.
struct ChunkedReaderStream {
    dtype: DType,
    pending_chunks: VecDeque<PendingChunk>,
    active_stream: Option<ReaderStreamRef>,
}

impl ChunkedReaderStream {
    /// Ensure we have an active stream pointing at the next non-exhausted chunk.
    fn ensure_active_stream(&mut self) -> VortexResult<bool> {
        loop {
            if let Some(ref stream) = self.active_stream {
                if stream.next_chunk_len().is_some() {
                    return Ok(true);
                }
                // Current stream is exhausted, drop it.
                self.active_stream = None;
            }

            // Try to activate the next pending chunk.
            let Some(pending) = self.pending_chunks.pop_front() else {
                return Ok(false);
            };
            self.active_stream = Some(pending.reader.execute(pending.row_range)?);
        }
    }
}

impl ReaderStream for ChunkedReaderStream {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn next_chunk_len(&self) -> Option<usize> {
        if let Some(ref stream) = self.active_stream {
            let len = stream.next_chunk_len();
            if len.is_some() {
                return len;
            }
        }
        // Peek at the next pending chunk's row count as a hint.
        self.pending_chunks
            .front()
            .map(|c| usize::try_from(c.row_range.end - c.row_range.start).unwrap_or(usize::MAX))
    }

    fn skip(&mut self, mut n: usize) {
        while n > 0 {
            // Try to skip within the active stream.
            if let Some(ref mut stream) = self.active_stream {
                if let Some(chunk_len) = stream.next_chunk_len() {
                    if n <= chunk_len {
                        stream.skip(n);
                        return;
                    }
                    stream.skip(chunk_len);
                    n -= chunk_len;
                }
                self.active_stream = None;
            }

            // Skip entire pending chunks without constructing streams.
            let Some(pending) = self.pending_chunks.front() else {
                vortex_panic!("Cannot skip {} more rows, no chunks remaining", n);
            };
            let chunk_rows = usize::try_from(pending.row_range.end - pending.row_range.start)
                .unwrap_or(usize::MAX);
            if n >= chunk_rows {
                self.pending_chunks.pop_front();
                n -= chunk_rows;
            } else {
                // Partial skip — construct the stream and skip within it.
                // We just checked front() is Some, so pop_front() is guaranteed.
                match self.pending_chunks.pop_front() {
                    Some(pending) => {
                        let mut stream = match pending.reader.execute(pending.row_range) {
                            Ok(s) => s,
                            Err(e) => vortex_panic!("failed to execute chunk reader: {e}"),
                        };
                        stream.skip(n);
                        self.active_stream = Some(stream);
                    }
                    None => vortex_panic!("pending chunk disappeared during skip"),
                }
                return;
            }
        }
    }

    fn next_chunk(
        &mut self,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        if !self.ensure_active_stream()? {
            vortex_bail!("No more chunks in chunked reader stream");
        }

        let stream = self
            .active_stream
            .as_mut()
            .ok_or_else(|| vortex_err!("Active stream missing after ensure"))?;
        let next_len = stream
            .next_chunk_len()
            .ok_or_else(|| vortex_err!("Active stream unexpectedly exhausted"))?;

        if mask.len() <= next_len {
            return stream.next_chunk(mask);
        }

        // Mask spans multiple chunks — gather results from each.
        let mut remaining_mask = mask;
        let mut futs = Vec::new();

        while !remaining_mask.is_empty() {
            if !self.ensure_active_stream()? {
                vortex_bail!("Ran out of chunks while processing mask");
            }

            let stream = self
                .active_stream
                .as_mut()
                .ok_or_else(|| vortex_err!("Active stream missing after ensure"))?;
            let chunk_len = stream
                .next_chunk_len()
                .ok_or_else(|| vortex_err!("Active stream unexpectedly exhausted"))?;

            let take = chunk_len.min(remaining_mask.len());
            let chunk_mask = remaining_mask.slice(0..take);
            remaining_mask = remaining_mask.slice(take..remaining_mask.len());

            futs.push(stream.next_chunk(chunk_mask)?);
        }

        let dtype = self.dtype.clone();
        Ok(async move {
            let arrays = try_join_all(futs).await?;
            Ok(ChunkedArray::try_new(arrays, dtype)?.into_array())
        }
        .boxed())
    }
}

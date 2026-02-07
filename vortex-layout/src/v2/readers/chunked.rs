// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;

use termtree::Tree;
use vortex_array::ArrayFuture;
use vortex_array::expr::Expression;
use vortex_dtype::DType;
use vortex_error::VortexResult;
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

    fn display_tree(&self) -> Tree<String> {
        let label = format!(
            "Chunked({}, rows={}, chunks={})",
            self.dtype,
            self.row_count,
            self.chunks.len()
        );
        let mut tree = Tree::new(label);
        for (i, chunk) in self.chunks.iter().enumerate() {
            let child = chunk.display_tree();
            tree.push(Tree::new(format!("[{}]: {}", i, child.root)).with_leaves(child.leaves));
        }
        tree
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
    /// Ensure we have an active stream pointing at a non-exhausted chunk.
    /// Returns `Ok(true)` if a stream is ready, `Ok(false)` if no more chunks.
    fn ensure_active_stream(&mut self) -> VortexResult<bool> {
        loop {
            if self.active_stream.is_some() {
                return Ok(true);
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

    fn skip(&mut self, mut n: usize) {
        while n > 0 {
            // Try to skip within the active stream.
            if let Some(ref mut stream) = self.active_stream {
                // We don't know the stream's remaining length without next_chunk_len,
                // so we try to skip and if there's no more data, move on.
                // For skip, the contract says n must not exceed remaining, so we delegate.
                stream.skip(n);
                return;
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

    fn next_chunk(&mut self) -> VortexResult<Option<ArrayFuture>> {
        loop {
            if !self.ensure_active_stream()? {
                return Ok(None);
            }

            let Some(stream) = self.active_stream.as_mut() else {
                return Ok(None);
            };

            match stream.next_chunk()? {
                Some(future) => return Ok(Some(future)),
                None => {
                    // Current stream is exhausted, try next chunk.
                    self.active_stream = None;
                }
            }
        }
    }
}

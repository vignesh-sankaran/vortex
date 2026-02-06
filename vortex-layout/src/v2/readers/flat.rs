// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::buffer::BufferHandle;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::expr::transform::replace;
use vortex_array::serde::ArrayParts;
use vortex_array::session::ArrayRegistry;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::SharedVortexResult;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::segments::SegmentId;
use crate::segments::SegmentSourceRef;
use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStream;
use crate::v2::reader::ReaderStreamRef;

type SharedSegmentFuture = Shared<BoxFuture<'static, SharedVortexResult<BufferHandle>>>;

/// A leaf reader that reads a single flat segment.
///
/// The segment source handles caching — multiple streams reading the same segment will share
/// the underlying I/O through the segment source's cache.
pub struct FlatReader {
    len: usize,
    dtype: DType,
    decode_dtype: DType,
    array_tree: Option<ByteBuffer>,
    segment_id: SegmentId,
    segment_source: SegmentSourceRef,
    ctx: ArrayContext,
    registry: ArrayRegistry,
    expression: Option<Expression>,
}

impl FlatReader {
    pub fn new(
        len: usize,
        dtype: DType,
        array_tree: Option<ByteBuffer>,
        segment_id: SegmentId,
        segment_source: SegmentSourceRef,
        ctx: ArrayContext,
        registry: ArrayRegistry,
    ) -> Self {
        Self {
            len,
            decode_dtype: dtype.clone(),
            dtype,
            array_tree,
            segment_id,
            segment_source,
            ctx,
            registry,
            expression: None,
        }
    }
}

impl Reader for FlatReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.len as u64
    }

    fn apply(&self, expression: &Expression) -> VortexResult<ReaderRef> {
        let new_expr = match &self.expression {
            None => expression.clone(),
            Some(existing) => replace(existing.clone(), &root(), expression.clone()),
        };
        let new_dtype = new_expr.return_dtype(&self.dtype)?;
        Ok(Arc::new(Self {
            len: self.len,
            dtype: new_dtype,
            decode_dtype: self.decode_dtype.clone(),
            array_tree: self.array_tree.clone(),
            segment_id: self.segment_id,
            segment_source: self.segment_source.clone(),
            ctx: self.ctx.clone(),
            registry: self.registry.clone(),
            expression: Some(new_expr),
        }))
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        let start = usize::try_from(row_range.start)
            .map_err(|_| vortex_err!("row range start too large for usize"))?;
        let end = usize::try_from(row_range.end)
            .map_err(|_| vortex_err!("row range end too large for usize"))?;

        if start > self.len || end > self.len || start > end {
            vortex_bail!(
                "Row range {}..{} out of bounds for flat reader of length {}",
                start,
                end,
                self.len
            );
        }

        // Request the segment once and share across all next_chunk calls.
        let segment_fut = self
            .segment_source
            .request(self.segment_id)
            .map(|r| r.map_err(Arc::new))
            .boxed()
            .shared();

        Ok(Box::new(FlatReaderStream {
            dtype: self.dtype.clone(),
            decode_dtype: self.decode_dtype.clone(),
            segment_fut,
            array_tree: self.array_tree.clone(),
            ctx: self.ctx.clone(),
            registry: self.registry.clone(),
            expression: self.expression.clone(),
            row_count: self.len,
            offset: start,
            remaining: end - start,
        }))
    }
}

struct FlatReaderStream {
    dtype: DType,
    decode_dtype: DType,
    segment_fut: SharedSegmentFuture,
    array_tree: Option<ByteBuffer>,
    ctx: ArrayContext,
    registry: ArrayRegistry,
    expression: Option<Expression>,
    row_count: usize,
    offset: usize,
    remaining: usize,
}

impl ReaderStream for FlatReaderStream {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn next_chunk_len(&self) -> Option<usize> {
        if self.remaining == 0 {
            None
        } else {
            Some(self.remaining)
        }
    }

    fn skip(&mut self, n: usize) {
        if n > self.remaining {
            vortex_panic!("Cannot skip {} rows, only {} remaining", n, self.remaining);
        }
        self.offset += n;
        self.remaining -= n;
    }

    fn next_chunk(
        &mut self,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        if mask.len() > self.remaining {
            vortex_bail!(
                "Mask length {} exceeds remaining rows {}",
                mask.len(),
                self.remaining
            );
        }

        let segment_fut = self.segment_fut.clone();
        let array_tree = self.array_tree.clone();
        let ctx = self.ctx.clone();
        let registry = self.registry.clone();
        let row_count = self.row_count;
        let offset = self.offset;
        let len = mask.len();
        let expression = self.expression.clone();
        let dtype = self.dtype.clone();
        let decode_dtype = self.decode_dtype.clone();

        self.offset += len;
        self.remaining -= len;

        Ok(async move {
            // Await the mask first — if all-false, skip I/O entirely.
            // TODO(ngates): should we race this against the segment future?
            let mask = mask.await?;
            if mask.true_count() == 0 {
                return Ok(Canonical::empty(&dtype).into_array());
            }

            // Await the shared segment future (I/O is issued once, shared across chunks).
            let segment = segment_fut.await?;
            let parts = if let Some(array_tree) = array_tree {
                // Use the pre-stored flatbuffer from layout metadata combined with segment buffers.
                ArrayParts::from_flatbuffer_and_segment(array_tree, segment)?
            } else {
                // Parse the flatbuffer from the segment itself.
                ArrayParts::try_from(segment)?
            };

            let mut array = parts.decode(&decode_dtype, row_count, &ctx, &registry)?;

            // Slice to the requested row range within the segment.
            if offset > 0 || len < row_count {
                array = array.slice(offset..offset + len)?;
            }

            // Filter using the mask.
            if !mask.all_true() {
                array = array.filter(mask)?;
            }

            // Apply any accumulated expression.
            if let Some(expr) = expression {
                array = array.apply(&expr)?;
            }

            Ok(array)
        }
        .boxed())
    }
}

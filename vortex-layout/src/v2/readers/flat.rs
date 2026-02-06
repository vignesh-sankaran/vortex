// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use termtree::Tree;
use vortex_array::ArrayContext;
use vortex_array::ArrayFuture;
use vortex_array::ArrayRef;
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

type SharedArrayFuture = Shared<BoxFuture<'static, SharedVortexResult<ArrayRef>>>;

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

        // Decode the array once and share the result across all next_chunk calls.
        let segment_source = self.segment_source.clone();
        let segment_id = self.segment_id;
        let array_tree = self.array_tree.clone();
        let decode_dtype = self.decode_dtype.clone();
        let row_count = self.len;
        let ctx = self.ctx.clone();
        let registry = self.registry.clone();
        let array_fut = async move {
            let segment = segment_source.request(segment_id).await?;
            let parts = if let Some(array_tree) = array_tree {
                ArrayParts::from_flatbuffer_and_segment(array_tree, segment)?
            } else {
                ArrayParts::try_from(segment)?
            };
            parts.decode(&decode_dtype, row_count, &ctx, &registry)
        }
        .map(|r| r.map_err(Arc::new))
        .boxed()
        .shared();

        Ok(Box::new(FlatReaderStream {
            dtype: self.dtype.clone(),
            array_fut,
            expression: self.expression.clone(),
            row_count: self.len,
            offset: start,
            remaining: end - start,
        }))
    }

    fn display_tree(&self) -> Tree<String> {
        let mut label = format!(
            "Flat({}, rows={}, segment={}",
            self.dtype, self.len, self.segment_id
        );
        if let Some(expr) = &self.expression {
            label.push_str(&format!(", expr={}", expr));
        }
        label.push(')');
        Tree::new(label)
    }
}

struct FlatReaderStream {
    dtype: DType,
    array_fut: SharedArrayFuture,
    expression: Option<Expression>,
    row_count: usize,
    offset: usize,
    remaining: usize,
}

impl ReaderStream for FlatReaderStream {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn skip(&mut self, n: usize) {
        if n > self.remaining {
            vortex_panic!("Cannot skip {} rows, only {} remaining", n, self.remaining);
        }
        self.offset += n;
        self.remaining -= n;
    }

    fn next_chunk(&mut self) -> Option<VortexResult<ArrayFuture>> {
        if self.remaining == 0 {
            return None;
        }

        let array_fut = self.array_fut.clone();
        let row_count = self.row_count;
        let offset = self.offset;
        let len = self.remaining;
        let expression = self.expression.clone();

        self.offset += len;
        self.remaining = 0;

        Some(Ok(ArrayFuture::new(len, async move {
            // Await the shared array future (decoded once, shared across chunks).
            let mut array: ArrayRef = array_fut.await?;

            // Slice to the requested row range within the segment.
            if offset > 0 || len < row_count {
                array = array.slice(offset..offset + len)?;
            }

            // Apply any accumulated expression.
            if let Some(expr) = expression {
                array = array.apply(&expr)?;
            }

            Ok(array)
        })))
    }
}

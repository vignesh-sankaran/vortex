// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use moka::future::FutureExt;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::expr::transform::replace;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

use crate::segments::SegmentId;
use crate::segments::SegmentSourceRef;
use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStream;
use crate::v2::reader::ReaderStreamRef;

/// A leaf reader that reads a single flat segment.
///
/// The segment source handles caching — multiple streams reading the same segment will share
/// the underlying I/O through the segment source's cache.
pub struct FlatReader {
    len: usize,
    dtype: DType,
    segment_id: SegmentId,
    segment_source: SegmentSourceRef,
    expression: Option<Expression>,
}

impl FlatReader {
    pub fn new(
        len: usize,
        dtype: DType,
        segment_id: SegmentId,
        segment_source: SegmentSourceRef,
    ) -> Self {
        Self {
            len,
            dtype,
            segment_id,
            segment_source,
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
            segment_id: self.segment_id,
            segment_source: self.segment_source.clone(),
            expression: Some(new_expr),
        }))
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        let start = usize::try_from(row_range.start)
            .map_err(|_| vortex_error::vortex_err!("row range start too large for usize"))?;
        let end = usize::try_from(row_range.end)
            .map_err(|_| vortex_error::vortex_err!("row range end too large for usize"))?;

        if start > self.len || end > self.len || start > end {
            vortex_bail!(
                "Row range {}..{} out of bounds for flat reader of length {}",
                start,
                end,
                self.len
            );
        }

        Ok(Box::new(FlatReaderStream {
            dtype: self.dtype.clone(),
            segment_id: self.segment_id,
            segment_source: self.segment_source.clone(),
            expression: self.expression.clone(),
            offset: start,
            remaining: end - start,
        }))
    }
}

struct FlatReaderStream {
    dtype: DType,
    segment_id: SegmentId,
    segment_source: SegmentSourceRef,
    expression: Option<Expression>,
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

        let segment_id = self.segment_id;
        let segments = self.segment_source.clone();
        let offset = self.offset;
        let len = mask.len();
        let expression = self.expression.clone();

        self.offset += len;
        self.remaining -= len;

        let dtype = self.dtype.clone();

        Ok(async move {
            // Await the mask first — if all-false, skip I/O entirely.
            let mask = mask.await?;
            if mask.true_count() == 0 {
                return Ok(Canonical::empty(&dtype).into_array());
            }

            // Issue I/O only after confirming we have rows to read.
            let _buffer = segments.request(segment_id).await?;
            // TODO(ngates): decode buffer into array using ArrayParts/ArrayContext,
            //  then slice + filter + apply expression.
            drop((offset, len, expression));
            todo!("decode segment buffer into array")
        }
        .boxed())
    }
}

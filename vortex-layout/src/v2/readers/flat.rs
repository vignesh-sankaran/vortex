// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::try_join;
use moka::future::FutureExt;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::expr::transform::replace;
use vortex_dtype::DType;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

use crate::layouts::SharedArrayFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSourceRef;
use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStream;
use crate::v2::reader::ReaderStreamRef;

pub struct FlatReader {
    len: usize,
    dtype: DType,
    segment_id: SegmentId,
    segment_source: SegmentSourceRef,
    expression: Option<Expression>,
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
        let new_dtype = expression.return_dtype(&self.dtype)?;
        Ok(match &self.expression {
            None => Arc::new(Self {
                len: self.len,
                dtype: new_dtype,
                segment_id: self.segment_id.clone(),
                segment_source: self.segment_source.clone(),
                expression: Some(expression.clone()),
            }),
            Some(e) => {
                let new_expr = replace(e.clone(), &root(), expression.clone());
                Arc::new(Self {
                    len: self.len,
                    dtype: new_dtype,
                    segment_id: self.segment_id.clone(),
                    segment_source: self.segment_source.clone(),
                    expression: Some(new_expr),
                })
            }
        })
    }

    fn execute(&self, _row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        todo!()
    }
}

struct FlatLayoutReaderStream {
    dtype: DType,
    array_fut: SharedArrayFuture,
    offset: usize,
    remaining: usize,
}

impl ReaderStream for FlatLayoutReaderStream {
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
                "Selection mask length {} exceeds remaining rows {}",
                mask.len(),
                self.remaining
            );
        }

        let array_fut = self.array_fut.clone();
        let offset = self.offset;
        let mask = mask.clone();

        self.offset += mask.len();
        self.remaining -= mask.len();

        Ok(async move {
            let (array, mask) = try_join!(array_fut.map_err(VortexError::from), mask)?;
            let sliced_array = array.slice(offset..offset + mask.len())?;
            let selected_array = sliced_array.filter(mask)?;
            Ok(selected_array)
        }
        .boxed())
    }
}

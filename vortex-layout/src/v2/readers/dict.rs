// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use termtree::Tree;
use vortex_array::ArrayFuture;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::SharedArray;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::expr::transform::replace;
use vortex_dtype::DType;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStream;
use crate::v2::reader::ReaderStreamRef;

type SharedVortexResult<T> = Result<T, Arc<VortexError>>;

/// A reader that reconstructs dict-encoded arrays from separate values and codes readers.
///
/// The values reader holds the dictionary (typically a small flat array), while the codes
/// reader holds integer indices into the dictionary (row-aligned with the parent layout).
pub struct DictReader {
    dtype: DType,
    values: ReaderRef,
    codes: ReaderRef,
    expression: Option<Expression>,
}

impl DictReader {
    pub fn new(dtype: DType, values: ReaderRef, codes: ReaderRef) -> Self {
        Self {
            dtype,
            values,
            codes,
            expression: None,
        }
    }
}

impl Reader for DictReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.codes.row_count()
    }

    fn apply(&self, expression: &Expression) -> VortexResult<ReaderRef> {
        // Dict encoding is transparent — expressions are applied after reconstruction.
        // We compose the expression and apply it after creating the DictArray.
        let new_expr = match &self.expression {
            None => expression.clone(),
            Some(existing) => replace(existing.clone(), &root(), expression.clone()),
        };
        let new_dtype = new_expr.return_dtype(&self.dtype)?;
        Ok(Arc::new(Self {
            dtype: new_dtype,
            values: self.values.clone(),
            codes: self.codes.clone(),
            expression: Some(new_expr),
        }))
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        // Read the full dictionary values once and share across all chunks.
        let values_row_count = self.values.row_count();
        let mut values_stream = self.values.execute(0..values_row_count)?;
        let values_array_future = values_stream
            .next_chunk()?
            .ok_or_else(|| vortex_err!("Dict values stream is empty"))?;
        let values_fut: Shared<BoxFuture<'static, SharedVortexResult<ArrayRef>>> = async move {
            let array = values_array_future.await?;
            Ok(SharedArray::new(array).into_array())
        }
        .map(|r: VortexResult<ArrayRef>| r.map_err(Arc::new))
        .boxed()
        .shared();

        let codes_stream = self.codes.execute(row_range)?;

        Ok(Box::new(DictReaderStream {
            dtype: self.dtype.clone(),
            codes_stream,
            values_fut,
            expression: self.expression.clone(),
            estimated_bytes_per_row: super::estimated_decoded_bytes(&self.dtype, 1),
        }))
    }

    fn display_tree(&self) -> Tree<String> {
        let mut label = format!("Dict({}, rows={}", self.dtype, self.codes.row_count());
        if let Some(expr) = &self.expression {
            label.push_str(&format!(", expr={}", expr));
        }
        label.push(')');

        let mut tree = Tree::new(label);

        let values_child = self.values.display_tree();
        tree.push(
            Tree::new(format!("values: {}", values_child.root)).with_leaves(values_child.leaves),
        );

        let codes_child = self.codes.display_tree();
        tree.push(
            Tree::new(format!("codes: {}", codes_child.root)).with_leaves(codes_child.leaves),
        );

        tree
    }
}

struct DictReaderStream {
    dtype: DType,
    codes_stream: ReaderStreamRef,
    values_fut: Shared<BoxFuture<'static, SharedVortexResult<ArrayRef>>>,
    expression: Option<Expression>,
    estimated_bytes_per_row: usize,
}

impl ReaderStream for DictReaderStream {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn skip(&mut self, n: usize) {
        self.codes_stream.skip(n);
    }

    fn next_chunk(&mut self) -> VortexResult<Option<ArrayFuture>> {
        let Some(codes_future) = self.codes_stream.next_chunk()? else {
            return Ok(None);
        };
        let values_fut = self.values_fut.clone();
        let expression = self.expression.clone();
        let len = codes_future.len();
        let estimated_bytes = len * self.estimated_bytes_per_row;

        Ok(Some(ArrayFuture::new(len, estimated_bytes, async move {
            let values = values_fut.await.map_err(|e| vortex_err!("{e}"))?;
            let codes = codes_future.await?;
            let mut array = DictArray::try_new(codes, values)?.into_array();
            if let Some(expr) = expression {
                array = array.apply(&expr)?;
            }
            Ok(array)
        })))
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use termtree::Tree;
use vortex_array::ArrayFuture;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::expr::transform::replace;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_scalar::Scalar;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStream;
use crate::v2::reader::ReaderStreamRef;

/// A reader that produces constant values.
pub struct ConstantReader {
    scalar: Scalar,
    row_count: u64,
    expression: Option<Expression>,
}

impl ConstantReader {
    pub fn new(scalar: Scalar, row_count: u64) -> Self {
        Self {
            scalar,
            row_count,
            expression: None,
        }
    }
}

impl Reader for ConstantReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.scalar.dtype()
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn apply(&self, expression: &Expression) -> VortexResult<ReaderRef> {
        let new_expr = match &self.expression {
            None => expression.clone(),
            Some(existing) => replace(existing.clone(), &root(), expression.clone()),
        };
        Ok(Arc::new(Self {
            scalar: self.scalar.clone(),
            row_count: self.row_count,
            expression: Some(new_expr),
        }))
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        let remaining = row_range.end.saturating_sub(row_range.start);
        Ok(Box::new(ConstantReaderStream {
            scalar: self.scalar.clone(),
            remaining,
            expression: self.expression.clone(),
        }))
    }

    fn display_tree(&self) -> Tree<String> {
        Tree::new(format!(
            "Constant({}, rows={}, value={})",
            self.scalar.dtype(),
            self.row_count,
            self.scalar
        ))
    }
}

struct ConstantReaderStream {
    scalar: Scalar,
    remaining: u64,
    expression: Option<Expression>,
}

impl ReaderStream for ConstantReaderStream {
    fn dtype(&self) -> &DType {
        self.scalar.dtype()
    }

    fn skip(&mut self, n: usize) {
        let n = n as u64;
        if n > self.remaining {
            vortex_panic!("Cannot skip {} rows, only {} remaining", n, self.remaining);
        }
        self.remaining -= n;
    }

    fn next_chunk(&mut self) -> Option<VortexResult<ArrayFuture>> {
        if self.remaining == 0 {
            return None;
        }

        let len = usize::try_from(self.remaining).unwrap_or(usize::MAX);
        let scalar = self.scalar.clone();
        let expression = self.expression.clone();
        self.remaining = 0;

        Some(Ok(ArrayFuture::new(len, async move {
            let mut array = ConstantArray::new(scalar, len).into_array();
            if let Some(e) = expression {
                array = array.apply(&e)?;
            }
            Ok(array)
        })))
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::future::try_join_all;
use itertools::Itertools;
use termtree::Tree;
use vortex_array::ArrayFuture;
use vortex_array::IntoArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::expr::Expression;
use vortex_array::expr::Literal;
use vortex_array::expr::Root;
use vortex_array::expr::ScalarFn;
use vortex_array::expr::VTable;
use vortex_array::expr::VTableExt;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStream;
use crate::v2::reader::ReaderStreamRef;
use crate::v2::readers::constant::ConstantReader;

/// A [`Reader`] for applying a scalar function to child readers.
pub struct ScalarFnReader {
    scalar_fn: ScalarFn,
    dtype: DType,
    row_count: u64,
    children: Vec<ReaderRef>,
}

impl ScalarFnReader {
    pub fn try_new(
        scalar_fn: ScalarFn,
        children: Vec<ReaderRef>,
        row_count: u64,
    ) -> VortexResult<Self> {
        let dtype = scalar_fn.return_dtype(
            &children
                .iter()
                .map(|c| c.dtype().clone())
                .collect::<Vec<DType>>(),
        )?;

        Ok(Self {
            scalar_fn,
            dtype,
            row_count,
            children,
        })
    }

    pub fn scalar_fn(&self) -> &ScalarFn {
        &self.scalar_fn
    }
}

impl Reader for ScalarFnReader {
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
        // Treat this ScalarFnReader as a data source. Resolve Root to a clone of self,
        // literals to constants, and wrap everything else in a new ScalarFnReader.
        if expression.is::<Root>() {
            return Ok(Arc::new(Self {
                scalar_fn: self.scalar_fn.clone(),
                dtype: self.dtype.clone(),
                row_count: self.row_count,
                children: self.children.clone(),
            }));
        }

        if let Some(scalar) = expression.as_opt::<Literal>() {
            return Ok(Arc::new(ConstantReader::new(
                scalar.clone(),
                self.row_count,
            )));
        }

        let resolved_children: Vec<ReaderRef> = expression
            .children()
            .iter()
            .map(|child| self.apply(child))
            .try_collect()?;

        Ok(Arc::new(Self::try_new(
            expression.scalar_fn().clone(),
            resolved_children,
            self.row_count,
        )?))
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        let input_streams = self
            .children
            .iter()
            .map(|child| child.execute(row_range.clone()))
            .collect::<VortexResult<Vec<_>>>()?;

        let num_inputs = input_streams.len();
        Ok(Box::new(ScalarFnArrayStream {
            dtype: self.dtype.clone(),
            scalar_fn: self.scalar_fn.clone(),
            input_streams,
            input_buffers: vec![None; num_inputs],
        }))
    }

    fn display_tree(&self) -> Tree<String> {
        let label = format!(
            "ScalarFn({}, rows={}, fn={})",
            self.dtype, self.row_count, self.scalar_fn
        );
        let mut tree = Tree::new(label);
        for (i, child) in self.children.iter().enumerate() {
            let child_tree = child.display_tree();
            tree.push(
                Tree::new(format!("[{}]: {}", i, child_tree.root)).with_leaves(child_tree.leaves),
            );
        }
        tree
    }
}

struct ScalarFnArrayStream {
    dtype: DType,
    scalar_fn: ScalarFn,
    input_streams: Vec<ReaderStreamRef>,
    input_buffers: Vec<Option<ArrayFuture>>,
}

impl ScalarFnArrayStream {
    /// Get the next ArrayFuture for an input stream, taking from the buffer first.
    fn next_for_input(
        stream: &mut ReaderStreamRef,
        buffer: &mut Option<ArrayFuture>,
    ) -> Option<VortexResult<ArrayFuture>> {
        if let Some(buffered) = buffer.take() {
            return Some(Ok(buffered));
        }
        stream.next_chunk()
    }
}

impl ReaderStream for ScalarFnArrayStream {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn skip(&mut self, n: usize) {
        for (stream, buffer) in self
            .input_streams
            .iter_mut()
            .zip(self.input_buffers.iter_mut())
        {
            let mut remaining = n;
            if let Some(buf) = buffer.take() {
                if remaining < buf.len() {
                    *buffer = Some(buf.slice(remaining..buf.len()));
                    continue;
                }
                remaining -= buf.len();
            }
            if remaining > 0 {
                stream.skip(remaining);
            }
        }
    }

    fn next_chunk(&mut self) -> Option<VortexResult<ArrayFuture>> {
        // Collect an ArrayFuture for each input.
        let mut all_futures: Vec<ArrayFuture> = Vec::with_capacity(self.input_streams.len());

        for (stream, buffer) in self
            .input_streams
            .iter_mut()
            .zip(self.input_buffers.iter_mut())
        {
            let future = match Self::next_for_input(stream, buffer) {
                Some(Ok(f)) => f,
                Some(Err(e)) => return Some(Err(e)),
                None => return None,
            };
            all_futures.push(future);
        }

        if all_futures.is_empty() {
            return None;
        }

        // Find the minimum length.
        let min_len = all_futures.iter().map(|f| f.len()).min().unwrap_or(0);
        if min_len == 0 {
            return None;
        }

        // For inputs with len > min_len, buffer the remainder and slice.
        let mut chunk_futures: Vec<ArrayFuture> = Vec::with_capacity(all_futures.len());
        for (idx, future) in all_futures.into_iter().enumerate() {
            if future.len() > min_len {
                let remainder = future.slice(min_len..future.len());
                let chunk = future.slice(0..min_len);
                self.input_buffers[idx] = Some(remainder);
                chunk_futures.push(chunk);
            } else {
                chunk_futures.push(future);
            }
        }

        let scalar_fn = self.scalar_fn.clone();
        Some(Ok(ArrayFuture::new(min_len, async move {
            let input_arrays = try_join_all(chunk_futures).await?;
            let array = ScalarFnArray::try_new(scalar_fn, input_arrays, min_len)?.into_array();
            let array = array.optimize()?;
            Ok(array)
        })))
    }
}

pub trait ScalarFnReaderExt: VTable {
    /// Creates a [`ScalarFnReader`] applying this scalar function to the given children.
    fn new_reader(
        &'static self,
        options: Self::Options,
        children: Vec<ReaderRef>,
        row_count: u64,
    ) -> VortexResult<ReaderRef> {
        Ok(Arc::new(ScalarFnReader::try_new(
            self.bind(options),
            children,
            row_count,
        )?))
    }
}
impl<V: VTable> ScalarFnReaderExt for V {}

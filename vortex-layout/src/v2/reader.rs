// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::expr::Expression;
use vortex_dtype::DType;
use vortex_error::VortexResult;

pub type ReaderRef = Arc<dyn Reader>;

/// A reader provides an interface for loading data from row-indexed layouts.
///
/// Readers have a concrete row count allowing fixed partitions over a known set of rows. Readers
/// are driven by asking for the next chunk size, before providing a [`MaskFuture`] that resolves
/// into a mask of that length.
pub trait Reader: 'static + Send + Sync {
    /// Downcast the reader to a concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Get the data type of the layout being read.
    fn dtype(&self) -> &DType;

    /// Returns the number of rows in the reader.
    fn row_count(&self) -> u64;

    /// Apply an expression to the reader, returning a new reader that will execute the expression
    /// on top of the current reader.
    fn apply(&self, expression: &Expression) -> VortexResult<ReaderRef>;

    /// Creates a scan over the given row range of the reader.
    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef>;
}

pub type ReaderStreamRef = Box<dyn ReaderStream>;

pub trait ReaderStream: 'static + Send + Sync {
    /// The data type of the returned data.
    fn dtype(&self) -> &DType;

    /// The preferred maximum row count for the next chunk.
    ///
    /// Returns [`None`] if there are no more chunks.
    fn next_chunk_len(&self) -> Option<usize>;

    /// Skip the next `n` rows of the stream.
    ///
    /// # Panics
    ///
    /// Panics if `n` is greater than the number of rows remaining in the stream..
    fn skip(&mut self, n: usize);

    /// Returns the next chunk of data given an input array.
    ///
    /// The returned chunk must have the same number of rows as the [`Mask::true_count`].
    /// The provided mask will have at most [`next_chunk_len`] rows.
    ///
    /// The returned future has a `'static` lifetime allowing the calling to drive the stream
    /// arbitrarily far without awaiting any data.
    fn next_chunk(
        &mut self,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>>;
}

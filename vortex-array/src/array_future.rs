// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayRef;

/// A future that resolves to an array with a known length.
#[derive(Clone)]
pub struct ArrayFuture {
    inner: Shared<BoxFuture<'static, SharedVortexResult<ArrayRef>>>,
    len: usize,
    estimated_bytes: usize,
}

impl ArrayFuture {
    /// Create a new `ArrayFuture` from a future that returns an array.
    pub fn new<F>(len: usize, estimated_bytes: usize, fut: F) -> Self
    where
        F: Future<Output = VortexResult<ArrayRef>> + Send + 'static,
    {
        Self {
            inner: fut
                .inspect(move |r| {
                    if let Ok(array) = r
                        && array.len() != len {
                            vortex_panic!("ArrayFuture created with future that returned array of incorrect length (expected {}, got {})", len, array.len());
                        }
                })
                .map_err(Arc::new)
                .boxed()
                .shared(),
            len,
            estimated_bytes,
        }
    }

    /// Create an `ArrayFuture` from an already-resolved array.
    pub fn ready(array: ArrayRef) -> Self {
        let len = array.len();
        Self::new(len, 0, async move { Ok(array) })
    }

    /// Returns the length of the array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the estimated decoded byte size of the array.
    pub fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }

    /// Create an `ArrayFuture` that resolves to a slice of the original array.
    pub fn slice(&self, range: Range<usize>) -> Self {
        let inner = self.inner.clone();
        let parent_len = self.len;
        let slice_estimated = if self.len > 0 {
            usize::try_from(self.estimated_bytes as u128 * range.len() as u128 / self.len as u128)
                .unwrap_or(usize::MAX)
        } else {
            0
        };
        Self::new(range.len(), slice_estimated, async move {
            let array = inner.await?;
            debug_assert!(range.end <= parent_len, "slice range out of bounds");
            let _ = parent_len;
            array.slice(range)
        })
    }
}

impl Future for ArrayFuture {
    type Output = VortexResult<ArrayRef>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        self.inner.poll_unpin(cx).map_err(VortexError::from)
    }
}

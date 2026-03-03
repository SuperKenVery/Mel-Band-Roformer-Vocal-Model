use burn::{
    module::{Module, AutodiffModule, ModuleVisitor, ModuleMapper, ModuleDisplay, ModuleDisplayDefault, DisplaySettings},
    tensor::backend::{Backend, AutodiffBackend},
    record::{Record, PrecisionSettings},
};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug)]
pub struct Ignored<T>(pub T);

impl<T: std::fmt::Debug> ModuleDisplayDefault for Ignored<T> {
    fn content(&self, content: burn::module::Content) -> Option<burn::module::Content> {
        Some(content)
    }
}

impl<T: std::fmt::Debug> ModuleDisplay for Ignored<T> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        None
    }
}

impl<T> Ignored<T> {
    pub fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T> std::ops::Deref for Ignored<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IgnoredRecord;

impl<B: Backend> Record<B> for IgnoredRecord {
    type Item<S: PrecisionSettings> = IgnoredRecord;
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        item
    }
}

impl<B: Backend, T: Clone + Send + Sync + std::fmt::Debug> Module<B> for Ignored<T> {
    type Record = IgnoredRecord;

    fn collect_devices(&self, _devices: burn::module::Devices<B>) -> Vec<B::Device> {
        Vec::new()
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn map<M: ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
        self
    }

    fn visit<V: ModuleVisitor<B>>(&self, _visitor: &mut V) {
        // Do nothing
    }

    fn into_record(self) -> Self::Record {
        IgnoredRecord
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }
}

impl<B: AutodiffBackend, T: Clone + Send + Sync + std::fmt::Debug> AutodiffModule<B> for Ignored<T> {
    type InnerModule = Ignored<T>;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }
}

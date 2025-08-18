# BitNet Training

[![Crates.io](https://img.shields.io/crates/v/bitnet-training.svg)](https://crates.io/crates/bitnet-training)
[![Documentation](https://docs.rs/bitnet-training/badge.svg)](https://docs.rs/bitnet-training)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)

Training and fine-tuning infrastructure for BitNet neural networks, providing quantization-aware training, parameter-efficient fine-tuning, and distributed training capabilities. **Currently implementing Phase 3: QAT (Quantization-Aware Training) Infrastructure** with straight-through estimator integration and comprehensive training workflows.

## 🎯 Purpose

`bitnet-training` provides comprehensive training infrastructure for BitNet models with **Phase 3: QAT Infrastructure implementation currently in active development**:

- **Quantization-Aware Training (QAT)**: Train models with quantization in the loop 🎯 **Currently Implementing**
- **Straight-Through Estimator**: Custom autograd functions for gradient flow through quantization 🎯 **Active Development**
- **QAT Optimizers**: Specialized optimizers adapted for quantized model training 🎯 **In Progress**
- **Error Analysis & Metrics**: Comprehensive quantization error monitoring and mitigation 🎯 **Active Development**
- **Training Monitoring**: Real-time metrics and visualization for QAT workflows 🎯 **In Progress**
- **Parameter-Efficient Fine-Tuning (PEFT)**: LoRA, QLoRA, and other efficient fine-tuning methods ⏳ **Planned**
- **Distributed Training**: Multi-GPU and multi-node training support ⏳ **Future**

## 🎯 Current Status: **Phase 3 QAT Infrastructure - ACTIVE DEVELOPMENT**

🎯 **This crate is currently implementing Phase 3: Quantization-Aware Training Infrastructure.**

### ✅ **Completed Foundation** (From Previous Phases)
- **BitLinear Layer Integration**: Complete BitLinear layer implementation ready for QAT integration
- **Memory Management Integration**: Full integration with bitnet-core's advanced memory management system  
- **Device Abstraction**: Seamless device-aware training across CPU/GPU platforms
- **Benchmarking Integration**: Ready for comprehensive QAT performance validation

### 🎯 **Phase 3: QAT Infrastructure** (Active Development - Weeks 5-6)

#### 🔄 **Currently Implementing**
- **Straight-Through Estimator**: Custom autograd functions for candle-core with gradient flow through quantization
- **QAT Loss Functions**: Quantization-aware loss functions with regularization terms
- **QAT Optimizers**: Adapted Adam/AdamW optimizers for quantized training workflows
- **Error Analysis System**: Real-time quantization error monitoring and layer-wise analysis
- **Training State Management**: QAT-specific checkpointing and resume functionality

#### ⏳ **Planned for Phase 3**
- **Progressive Quantization**: Layer-wise quantization scheduling and gradual quantization increase
- **Knowledge Distillation**: Teacher-student training for better quantized model quality  
- **Integration Testing**: Comprehensive integration with BitLinear layers and calibration system
- **Memory-Efficient Training**: QAT training with existing memory pools and optimization
- **Production Examples**: Complete QAT training workflows and usage examples

## 🚀 Phase 3 QAT Implementation Status & Timeline

### Current QAT Infrastructure Implementation Progress

| Component | Status | Progress | Target Completion |
|-----------|--------|----------|------------------|
| **Straight-Through Estimator** | 🔄 In Progress | 40% | Week 5 |
| **Custom Autograd Functions** | 🔄 In Progress | 30% | Week 5 |
| **QAT Loss Functions** | 🔄 In Progress | 25% | Week 5-6 |
| **QAT Optimizers** | ⏳ Planned | 0% | Week 6 |
| **Error Analysis & Metrics** | 🔄 In Progress | 35% | Week 5-6 |
| **Training State Management** | ⏳ Planned | 0% | Week 6 |

### Phase 3 Development Priorities

**Week 5 Focus (Current):**
- [ ] Complete straight-through estimator with gradient flow
- [ ] Implement custom autograd functions for candle-core
- [ ] Create QAT-specific loss functions with regularization
- [ ] Build comprehensive error analysis and metrics system

**Week 6 Focus:**
- [ ] Implement quantization-aware optimizer adaptations
- [ ] Add progressive quantization scheduling system
- [ ] Create knowledge distillation training infrastructure
- [ ] Build comprehensive training state tracking and checkpointing

**Integration & Production:**
- [ ] Complete integration testing with BitLinear layers
- [ ] Validate memory-efficient QAT training workflows  
- [ ] Create production-ready examples and documentation
- [ ] Performance benchmarking and optimization validation

## � QAT Implementation Details

### ✅ **Quantization-Aware Training** (Active Implementation)

#### Straight-Through Estimator (In Progress)
- **Forward Pass Quantization**: ✅ Simulate quantization during forward pass with BitLinear integration
- **Gradient Flow**: 🔄 Implement straight-through estimator for gradient estimation through quantization
- **Custom Autograd**: 🔄 Custom autograd functions for candle-core with quantization operations
- **STE Variants**: ⏳ Multiple STE variants (clipped, soft, learnable) for different use cases

#### QAT Training Infrastructure (In Progress)  
- **QAT Loss Functions**: 🔄 Quantization-aware loss functions with regularization terms
- **QAT Optimizers**: ⏳ Adam/AdamW optimizers adapted for quantized training workflows
- **Progressive Quantization**: ⏳ Gradually increase quantization during training for better convergence
- **Training Scheduling**: ⏳ Learning rate schedules optimized for QAT workflows

#### Error Analysis & Metrics (Active Development)
- **Real-time Error Monitoring**: 🔄 Track quantization errors during training with layer-wise analysis
- **SQNR Metrics**: 🔄 Signal-to-Quantization-Noise Ratio calculations for quality assessment
- **Layer Sensitivity**: 🔄 Layer-wise sensitivity analysis for mixed-precision decisions  
- **Error Mitigation**: ⏳ Automatic error mitigation strategies based on metrics

### ⏳ **Advanced Training Features** (Planned for Future Phases)

#### Parameter-Efficient Fine-Tuning
- **LoRA (Low-Rank Adaptation)**: Implement LoRA adaptation layers with rank selection
- **QLoRA (Quantized LoRA)**: Fine-tune 4-bit quantized base models with memory efficiency
- **Advanced PEFT Methods**: Prefix tuning, P-Tuning v2, AdaLoRA, and BitFit implementations

#### Model Parallelism
- **Pipeline Parallelism**: Split model across devices by layers
- **Tensor Parallelism**: Split individual layers across devices
- **Hybrid Parallelism**: Combine data and model parallelism
- **Memory Optimization**: Optimize memory usage in distributed setting

#### Communication
- **AllReduce Operations**: Efficient gradient reduction
- **Communication Backends**: Support for NCCL, Gloo, and MPI
- **Compression**: Gradient compression for reduced communication
- **Fault Tolerance**: Handle device failures gracefully

## 🚀 Planned API Design

### Basic Training

```rust
use bitnet_training::{Trainer, TrainingConfig, QATConfig};
use bitnet_core::{Tensor, Device};
use bitnet_quant::BitNetQuantizer;

// Configure quantization-aware training
let qat_config = QATConfig {
    enable_weight_quantization: true,
    enable_activation_quantization: true,
    quantization_schedule: QuantizationSchedule::Progressive,
    fake_quantization: true,
};

// Configure training
let config = TrainingConfig {
    learning_rate: 1e-4,
    batch_size: 32,
    num_epochs: 10,
    optimizer: OptimizerType::AdamW,
    qat_config: Some(qat_config),
    ..Default::default()
};

// Create trainer
let trainer = Trainer::new(model, config)?;

// Train model
let training_results = trainer.train(train_dataset, val_dataset).await?;
```

### LoRA Fine-Tuning

```rust
use bitnet_training::{LoRAConfig, LoRATrainer, PEFTConfig};

// Configure LoRA
let lora_config = LoRAConfig {
    rank: 16,
    alpha: 32,
    dropout: 0.1,
    target_modules: vec!["q_proj", "v_proj", "k_proj", "o_proj"],
    bias: BiasType::None,
};

// Configure PEFT training
let peft_config = PEFTConfig {
    method: PEFTMethod::LoRA(lora_config),
    base_model_quantization: Some(QuantizationConfig::int4()),
    gradient_checkpointing: true,
    ..Default::default()
};

// Create LoRA trainer
let lora_trainer = LoRATrainer::new(base_model, peft_config)?;

// Fine-tune with LoRA
let fine_tuned_model = lora_trainer.fine_tune(
    fine_tuning_dataset,
    validation_dataset
).await?;
```

### Distributed Training

```rust
use bitnet_training::{DistributedTrainer, DistributedConfig, ParallelismStrategy};

// Configure distributed training
let distributed_config = DistributedConfig {
    world_size: 4,
    rank: 0,
    backend: CommunicationBackend::NCCL,
    strategy: ParallelismStrategy::DataParallel,
    gradient_compression: true,
};

// Create distributed trainer
let distributed_trainer = DistributedTrainer::new(
    model,
    training_config,
    distributed_config
)?;

// Initialize distributed training
distributed_trainer.init_process_group().await?;

// Train with multiple GPUs
let results = distributed_trainer.train(dataset).await?;
```

### Advanced Training Features

```rust
use bitnet_training::{
    TrainingCallbacks, MetricsLogger, CheckpointManager,
    GradientAccumulator, MixedPrecisionTrainer
};

// Set up training callbacks
let callbacks = TrainingCallbacks::builder()
    .add_early_stopping(patience=5, metric="val_loss")
    .add_lr_scheduler(SchedulerType::CosineAnnealing)
    .add_model_checkpointing(save_every=1000)
    .add_metrics_logging(log_every=100)
    .build();

// Configure mixed precision training
let mixed_precision_config = MixedPrecisionConfig {
    enabled: true,
    loss_scale: LossScale::Dynamic,
    growth_factor: 2.0,
    backoff_factor: 0.5,
};

// Create advanced trainer
let trainer = Trainer::builder()
    .model(model)
    .config(training_config)
    .callbacks(callbacks)
    .mixed_precision(mixed_precision_config)
    .gradient_accumulation_steps(4)
    .build()?;

// Train with all features
let results = trainer.train(train_data, val_data).await?;
```

## 🏗️ Planned Architecture

### Core Components

```
bitnet-training/src/
├── lib.rs                   # Main library interface
├── trainer/                 # Core training infrastructure
│   ├── mod.rs              # Trainer interface
│   ├── base_trainer.rs     # Base trainer implementation
│   ├── qat_trainer.rs      # Quantization-aware trainer
│   ├── peft_trainer.rs     # PEFT trainer
│   ├── distributed_trainer.rs # Distributed trainer
│   └── callbacks.rs        # Training callbacks
├── qat/                     # Quantization-aware training
│   ├── mod.rs              # QAT interface
│   ├── fake_quantization.rs # Fake quantization implementation
│   ├── straight_through.rs # Straight-through estimator
│   ├── noise_injection.rs  # Quantization noise
│   ├── progressive.rs      # Progressive quantization
│   └── calibration.rs      # Online calibration
├── peft/                    # Parameter-efficient fine-tuning
│   ├── mod.rs              # PEFT interface
│   ├── lora/               # LoRA implementation
│   │   ├── mod.rs          # LoRA interface
│   │   ├── layers.rs       # LoRA layers
│   │   ├── config.rs       # LoRA configuration
│   │   ├── merging.rs      # LoRA weight merging
│   │   └── scaling.rs      # LoRA scaling strategies
│   ├── qlora/              # QLoRA implementation
│   │   ├── mod.rs          # QLoRA interface
│   │   ├── quantization.rs # 4-bit quantization
│   │   ├── paged_optimizer.rs # Paged optimizers
│   │   └── double_quant.rs # Double quantization
│   ├── prefix_tuning/      # Prefix tuning
│   ├── p_tuning/           # P-tuning v2
│   └── adalora/            # AdaLoRA
├── optimizers/              # Training optimizers
│   ├── mod.rs              # Optimizer interface
│   ├── qat_sgd.rs          # QAT-aware SGD
│   ├── qat_adam.rs         # QAT-aware Adam
│   ├── paged_adamw.rs      # Paged AdamW for large models
│   ├── schedulers.rs       # Learning rate schedulers
│   └── gradient_clipping.rs # Gradient clipping
├── distributed/             # Distributed training
│   ├── mod.rs              # Distributed interface
│   ├── data_parallel.rs    # Data parallelism
│   ├── model_parallel.rs   # Model parallelism
│   ├── pipeline_parallel.rs # Pipeline parallelism
│   ├── communication.rs    # Communication primitives
│   ├── synchronization.rs  # Gradient synchronization
│   └── fault_tolerance.rs  # Fault tolerance
├── data/                    # Data loading and processing
│   ├── mod.rs              # Data interface
│   ├── dataset.rs          # Dataset implementations
│   ├── dataloader.rs       # Data loading utilities
│   ├── preprocessing.rs    # Data preprocessing
│   ├── augmentation.rs     # Data augmentation
│   └── streaming.rs        # Streaming datasets
├── loss/                    # Loss functions
│   ├── mod.rs              # Loss interface
│   ├── language_modeling.rs # Language modeling loss
│   ├── contrastive.rs      # Contrastive learning
│   ├── distillation.rs     # Knowledge distillation
│   └── regularization.rs   # Regularization terms
├── metrics/                 # Training metrics
│   ├── mod.rs              # Metrics interface
│   ├── accuracy.rs         # Accuracy metrics
│   ├── perplexity.rs       # Perplexity calculation
│   ├── bleu.rs             # BLEU score
│   └── custom.rs           # Custom metrics
├── checkpointing/           # Model checkpointing
│   ├── mod.rs              # Checkpointing interface
│   ├── checkpoint_manager.rs # Checkpoint management
│   ├── state_dict.rs       # State dictionary handling
│   ├── resume.rs           # Training resumption
│   └── versioning.rs       # Checkpoint versioning
├── monitoring/              # Training monitoring
│   ├── mod.rs              # Monitoring interface
│   ├── logger.rs           # Training logger
│   ├── tensorboard.rs      # TensorBoard integration
│   ├── wandb.rs            # Weights & Biases integration
│   └── profiler.rs         # Training profiler
└── utils/                   # Training utilities
    ├── mod.rs              # Utility interface
    ├── memory.rs           # Memory management utilities
    ├── reproducibility.rs  # Reproducibility helpers
    ├── validation.rs       # Training validation
    └── debugging.rs        # Training debugging tools
```

### Integration Architecture

```rust
// Integration with other BitNet crates
use bitnet_core::memory::HybridMemoryPool;
use bitnet_quant::BitNetQuantizer;
use bitnet_metal::MetalDevice;
use bitnet_inference::InferenceEngine;

// Unified training pipeline
let pool = HybridMemoryPool::new()?;
let quantizer = BitNetQuantizer::new(qat_config.quantization)?;
let device = MetalDevice::default()?;

let trainer = Trainer::builder()
    .memory_pool(pool)
    .quantizer(quantizer)
    .device(device)
    .inference_engine(inference_engine) // For validation
    .build()?;
```

## 📊 Expected Performance Characteristics

### Training Performance (Projected)

| Model Size | Batch Size | GPU Memory | Training Speed | Convergence |
|------------|------------|------------|----------------|-------------|
| **7B params** | 8 | 24 GB | 1.2 steps/s | 2x faster |
| **7B params** | 16 | 40 GB | 0.8 steps/s | 2x faster |
| **13B params** | 4 | 24 GB | 0.6 steps/s | 1.8x faster |
| **13B params** | 8 | 48 GB | 0.4 steps/s | 1.8x faster |

### Memory Efficiency

| Training Method | Memory Usage | Trainable Params | Performance |
|----------------|--------------|------------------|-------------|
| **Full Fine-tuning** | 100% | 100% | 100% |
| **LoRA (r=16)** | 25% | 0.1% | 95% |
| **QLoRA (4-bit)** | 15% | 0.1% | 93% |
| **BitNet QAT** | 30% | 100% | 98% |

### Distributed Training Scaling

| GPUs | Scaling Efficiency | Communication Overhead | Memory per GPU |
|------|-------------------|------------------------|----------------|
| **1** | 100% | 0% | 24 GB |
| **2** | 95% | 5% | 12 GB |
| **4** | 90% | 10% | 6 GB |
| **8** | 85% | 15% | 3 GB |

## 🧪 Planned Testing Strategy

### Unit Tests
```bash
# Test QAT components
cargo test --package bitnet-training qat

# Test PEFT methods
cargo test --package bitnet-training peft

# Test optimizers
cargo test --package bitnet-training optimizers

# Test distributed training
cargo test --package bitnet-training distributed
```

### Integration Tests
```bash
# Test end-to-end training
cargo test --package bitnet-training --test e2e_training

# Test model convergence
cargo test --package bitnet-training --test convergence

# Test distributed training
cargo test --package bitnet-training --test distributed_training
```

### Performance Tests
```bash
# Benchmark training performance
cargo bench --package bitnet-training -- training

# Memory usage benchmarks
cargo bench --package bitnet-training -- memory

# Distributed scaling benchmarks
cargo bench --package bitnet-training -- scaling
```

### Accuracy Tests
```bash
# Validate QAT accuracy
cargo test --package bitnet-training --test qat_accuracy

# Validate PEFT accuracy
cargo test --package bitnet-training --test peft_accuracy

# Compare with baseline models
cargo test --package bitnet-training --test baseline_comparison
```

## 🔧 Configuration

### Training Configuration

```rust
use bitnet_training::{TrainingConfig, OptimizerConfig, SchedulerConfig};

let config = TrainingConfig {
    // Basic training parameters
    learning_rate: 1e-4,
    batch_size: 32,
    num_epochs: 10,
    max_steps: None,
    
    // Optimizer configuration
    optimizer: OptimizerConfig::AdamW {
        weight_decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    },
    
    // Learning rate scheduler
    scheduler: SchedulerConfig::CosineAnnealing {
        t_max: 10000,
        eta_min: 1e-6,
        warmup_steps: 1000,
    },
    
    // Regularization
    gradient_clip_norm: Some(1.0),
    dropout: 0.1,
    weight_decay: 0.01,
    
    // Checkpointing
    save_every: 1000,
    save_total_limit: 5,
    save_best_only: true,
    
    // Validation
    eval_every: 500,
    eval_steps: 100,
    early_stopping_patience: 5,
    
    // Mixed precision
    mixed_precision: true,
    gradient_accumulation_steps: 4,
    
    // Logging
    log_every: 100,
    log_level: LogLevel::Info,
};
```

### QAT Configuration

```rust
use bitnet_training::{QATConfig, QuantizationSchedule, FakeQuantConfig};

let qat_config = QATConfig {
    // Quantization settings
    weight_quantization: QuantizationConfig {
        bits: 1.58,
        symmetric: true,
        per_channel: false,
    },
    
    activation_quantization: QuantizationConfig {
        bits: 8,
        symmetric: false,
        per_channel: false,
    },
    
    // Training schedule
    quantization_schedule: QuantizationSchedule::Progressive {
        start_epoch: 2,
        end_epoch: 8,
        schedule_type: ScheduleType::Linear,
    },
    
    // Fake quantization
    fake_quantization: FakeQuantConfig {
        enabled: true,
        noise_factor: 0.1,
        straight_through_estimator: true,
    },
    
    // Calibration
    online_calibration: true,
    calibration_frequency: 1000,
    calibration_samples: 512,
};
```

## 🚀 Research Implementation

### Quantization-Aware Training

QAT for BitNet involves several key innovations:

1. **Straight-Through Estimator**: Gradient estimation through discrete quantization
2. **Progressive Quantization**: Gradually increase quantization during training
3. **Noise Injection**: Add quantization noise to improve robustness
4. **Online Calibration**: Update quantization parameters during training

### Parameter-Efficient Fine-Tuning

PEFT methods reduce the number of trainable parameters:

1. **LoRA**: Low-rank adaptation with minimal parameters
2. **QLoRA**: Combine LoRA with 4-bit quantization
3. **Prefix Tuning**: Learn task-specific prefixes
4. **AdaLoRA**: Adaptive budget allocation for LoRA

## 🤝 Contributing

This crate needs complete implementation! Priority areas:

1. **QAT Implementation**: Build quantization-aware training infrastructure
2. **PEFT Methods**: Implement LoRA, QLoRA, and other PEFT techniques
3. **Training Loop**: Create flexible and efficient training loops
4. **Distributed Training**: Add multi-GPU and multi-node support

### Getting Started

1. Study quantization-aware training literature
2. Implement basic training loop with fake quantization
3. Add LoRA implementation for parameter-efficient fine-tuning
4. Implement gradient synchronization for distributed training
5. Add comprehensive benchmarks and accuracy tests

### Development Priorities

## 🎯 Phase 3 Success Criteria & Integration

### Technical Targets for Phase 3 Completion

**QAT Training System:**
- ✅ Straight-through estimator with stable gradient flow through quantization layers
- ✅ Custom autograd functions seamlessly integrated with candle-core
- ✅ Quantization-aware loss functions with effective regularization terms
- ✅ Progressive quantization scheduling for optimal convergence
- ✅ Knowledge distillation integration for better quantized model quality

**Error Analysis & Metrics:**
- ✅ Real-time quantization error monitoring during training with <5% overhead
- ✅ Layer-wise sensitivity analysis for mixed-precision decision making
- ✅ Comprehensive SQNR and error visualization tools for debugging
- ✅ Automated error mitigation strategies based on configurable thresholds
- ✅ Integration with bitnet-quant calibration system for optimal parameters

**Performance & Memory Targets:**
- Training Overhead: <20% slowdown vs full-precision training
- Memory Efficiency: 60-70% memory reduction during QAT training
- Convergence Quality: Maintain model accuracy within 2% of full-precision baseline
- Integration: Seamless operation with existing BitLinear layers and memory management
- Production Ready: Comprehensive error handling and checkpointing system

### Integration with BitNet Ecosystem

**Calibration Integration** (with bitnet-quant):
- Seamless integration with streaming calibration dataset system
- Automatic quantization parameter optimization from calibration statistics
- Memory-efficient calibration-to-training pipeline workflows

**BitLinear Integration** (with bitnet-quant):
- Direct integration with completed Phase 2 BitLinear layer implementation
- QAT training workflows optimized for BitLinear quantization patterns
- Performance validation using established BitLinear benchmarking suite

**Memory Integration** (with bitnet-core):
- QAT training workflows using existing HybridMemoryPool system
- Device-aware training with automatic device selection and optimization
- Memory-efficient training with automatic cleanup and compaction

## 🚀 Development Roadmap

### Current Phase 3 (Weeks 5-6): QAT Infrastructure
1. **Straight-Through Estimator**: Complete STE implementation with gradient flow
2. **QAT Training System**: Loss functions, optimizers, and progressive quantization
3. **Error Analysis**: Comprehensive metrics and real-time monitoring
4. **Integration Testing**: Validation with BitLinear layers and memory management

### Future Phases (Post Phase 3):
1. **Phase 4**: Model architecture integration and transformer quantization
2. **Phase 5**: Advanced PEFT methods (LoRA, QLoRA, AdaLoRA)
3. **Phase 6**: Distributed training and production optimization
4. **Phase 7**: CLI tools and deployment infrastructure

## 📚 References

- **QAT Survey**: [Quantization Aware Training: A Survey](https://arxiv.org/abs/2004.04395)
- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **QLoRA Paper**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **BitNet Paper**: [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453)
- **Distributed Training**: [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

## 📄 License

Licensed under the MIT License. See [LICENSE](../LICENSE) for details.
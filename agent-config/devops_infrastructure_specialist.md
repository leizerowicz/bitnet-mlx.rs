# BitNet-Rust DevOps & Infrastructure Specialist - Docker BitNet Swarm Intelligence Infrastructure

> **⚠️ MANDATORY ORCHESTRATOR ROUTING**: Before executing any work from this specialist config, **ALWAYS consult `agent-config/orchestrator.md` FIRST** for task routing, workflow coordination, multi-agent needs, current project context, and agent hooks integration. The orchestrator serves as the central command that knows when and how to use this specialist.

> **Last Updated**: October 6, 2025 - **Docker BitNet Swarm Intelligence Phase** - Implementing production-ready Docker container infrastructure with perfect technical foundation (100% test success rate)

## Docker Container Integration
- **Container Role**: **PRIMARY** infrastructure and deployment agent in BitNet swarm - responsible for Docker container infrastructure and production deployment
- **API Endpoints**: `/api` (infrastructure monitoring), `/agents/devops/deploy` (deployment), `/agents/devops/monitor` (monitoring)
- **MCP Tools**: `docker-management`, `container-orchestration`, `infrastructure-monitoring`, `deployment-automation`, `performance-monitoring`
- **Resource Requirements**: Medium CPU, Medium memory for infrastructure management and deployment automation
- **Coordination Patterns**: **Swarm Mode**: Independent infrastructure management with collaborative optimization. **Hive Mind Mode**: Unified infrastructure approach for enterprise-scale deployment

## 🎯 DOCKER BITNET SWARM INTELLIGENCE INFRASTRUCTURE CAPABILITIES

### 🐝 Swarm Intelligence Infrastructure (Diverging Collaborative Operations)
**Use Cases for DevOps in Swarm Mode**:
- **Multi-Environment Deployment**: Different agents manage different environments (dev, staging, prod), then collaborative coordination
- **Infrastructure Optimization**: Independent optimization of different infrastructure components, then collaborative resource allocation
- **Deployment Strategy Exploration**: Different agents explore different deployment approaches, then consensus on optimal strategies
- **Performance Monitoring**: Parallel monitoring of different system components, then collaborative performance analysis

### 🧠 Hive Mind Intelligence Infrastructure (Unified Collective Operations)
**Use Cases for DevOps in Hive Mind Mode**:
- **Enterprise-Scale Deployment**: All agents work with unified deployment strategy across entire infrastructure
- **System-Wide Infrastructure Management**: Coordinated infrastructure management with unified operational procedures
- **Large-Scale Container Orchestration**: Unified approach to managing complex container orchestration systems
- **Production Incident Response**: Coordinated incident response with unified operational protocols

## Role Overview
You are the DevOps & Infrastructure Specialist for BitNet-Rust, responsible for designing, implementing, and maintaining the **Docker BitNet Swarm Intelligence container infrastructure** that supports VS Code extension integration with enterprise-grade reliability, security, and scalability.

## Project Context
BitNet-Rust has achieved perfect technical foundation with 100% test success rate and is implementing Docker containerized intelligence systems with production-ready deployment capabilities.

**Current Status**: 🎯 **DOCKER BITNET SWARM INTELLIGENCE INFRASTRUCTURE** - Container Production Deployment (October 6, 2025)
- **Foundation Complete**: 100% test success rate (1,169/1,169 tests passing) with perfect stability ✅
- **Performance Validated**: ARM64 NEON optimization achieving 1.37x-3.20x speedup (Microsoft parity complete)
- **Current Priority**: Production-ready Docker container with swarm/hive mind intelligence systems
- **Infrastructure Focus**: Container orchestration, VS Code integration, and operational excellence

## 🎯 Docker BitNet Swarm Container Infrastructure

### Production Container Architecture
**Multi-Stage Docker Build System** with ARM64/AMD64 multi-arch support:

```dockerfile
# Multi-stage production build for BitNet Swarm Intelligence
FROM rust:1.75-slim AS builder
# Build stage with ARM64 NEON optimizations preserved
RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

FROM debian:bookworm-slim AS runtime  
# Minimal runtime with essential dependencies only
RUN apt-get update && apt-get install -y \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

FROM runtime AS model-cache
# Pre-download microsoft/bitnet-b1.58-2B-4T-gguf model
COPY --from=builder /usr/local/cargo/bin/bitnet-cli /usr/local/bin/
RUN bitnet-cli model download microsoft/bitnet-b1.58-2B-4T-gguf

FROM model-cache AS swarm-intelligence
# BitNet Swarm Intelligence runtime with all systems
COPY swarm-orchestrator /usr/local/bin/
COPY api-server /usr/local/bin/
COPY mcp-server /usr/local/bin/
EXPOSE 8080 3000
CMD ["swarm-orchestrator", "--api-port", "8080", "--mcp-port", "3000"]
```

### Container Orchestration & Deployment
**Production-Ready Container System**:
- **Multi-Architecture Support**: ARM64 (Apple Silicon) and AMD64 (Intel/AMD) with optimized builds
- **Model Caching**: Pre-loaded microsoft/bitnet-b1.58-2B-4T-gguf model for instant startup
- **Service Discovery**: Automatic agent discovery and coordination within container
- **Health Monitoring**: Container health checks and automatic restart capabilities
- **Resource Optimization**: <2GB total image size with model included

### VS Code Extension Integration Infrastructure
**Container-to-Extension Communication**:
- **HTTP API Server**: Universal `/api` endpoint running on port 8080
- **MCP Server Integration**: Agent-aware MCP server on port 3000
- **Real-time Communication**: WebSocket support for live code generation
- **Container Networking**: Optimized networking for VS Code extension integration

## Core Responsibilities

### 1. Production Infrastructure Design & Management
- **Multi-Cloud Strategy**: Primary AWS, disaster recovery on Azure/GCP
- **High Availability**: Multi-region deployment with 99.9% uptime SLA
- **Auto-Scaling**: Kubernetes-based horizontal and vertical scaling
- **Disaster Recovery**: RTO <15 minutes, RPO <5 minutes

### 2. Container Orchestration & Deployment
- **Kubernetes Management**: EKS cluster management with best practices
- **CI/CD Pipeline**: Automated testing, building, and deployment
- **Blue-Green Deployment**: Zero-downtime deployments with rollback capability
- **Resource Management**: CPU, memory, and GPU resource optimization

### 3. Monitoring & Observability
- **Application Monitoring**: Response times, error rates, throughput metrics
- **Infrastructure Monitoring**: System health, resource utilization, capacity planning
- **Logging**: Centralized log management with structured logging
- **Alerting**: Proactive alerting with escalation procedures

### 4. Security & Compliance
- **Network Security**: VPC isolation, security groups, WAF protection
- **Secret Management**: Secure handling of API keys, certificates, and credentials
- **Compliance**: SOC 2, GDPR, and security audit readiness
- **Vulnerability Management**: Regular security scanning and patch management

## Infrastructure Architecture

### Multi-Cloud Production Architecture
```mermaid
graph TB
    subgraph "Global Load Balancer"
        CF[CloudFlare]
        WAF[Web Application Firewall]
    end
    
    subgraph "Primary Region - AWS US-East-1"
        ALB[Application Load Balancer]
        EKS[EKS Cluster]
        RDS[RDS PostgreSQL Multi-AZ]
        REDIS[ElastiCache Redis]
        S3[S3 + CloudFront]
    end
    
    subgraph "DR Region - AWS EU-West-1"
        ALB2[Application Load Balancer]
        EKS2[EKS Cluster]
        RDS2[RDS Read Replica]
        REDIS2[ElastiCache Redis]
        S3_DR[S3 Cross-Region Replication]
    end
    
    subgraph "Monitoring & Operations"
        PROM[Prometheus]
        GRAF[Grafana]
        ELK[ELK Stack]
        PD[PagerDuty]
    end
    
    CF --> ALB
    CF --> ALB2
    ALB --> EKS
    EKS --> RDS
    EKS --> REDIS
    EKS --> S3
```

### Kubernetes Cluster Configuration
```yaml
# Production EKS cluster configuration
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: bitnet-prod-cluster
  region: us-east-1
  version: "1.27"

nodeGroups:
  - name: system-nodes
    instanceType: t3.large
    minSize: 3
    maxSize: 6
    desiredCapacity: 3
    volumeSize: 100
    ssh:
      allow: false
    iam:
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
        albIngress: true
        
  - name: app-nodes
    instanceType: c5.2xlarge
    minSize: 3
    maxSize: 20
    desiredCapacity: 6
    volumeSize: 100
    labels:
      workload: application
    taints:
      - key: workload
        value: application
        effect: NoSchedule
        
  - name: gpu-nodes
    instanceType: p3.2xlarge
    minSize: 0
    maxSize: 10
    desiredCapacity: 2
    volumeSize: 200
    labels:
      workload: gpu
    taints:
      - key: workload
        value: gpu
        effect: NoSchedule

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-load-balancer-controller
  - name: cluster-autoscaler
  - name: metrics-server
```

### Application Deployment Configuration
```yaml
# BitNet API Gateway Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bitnet-api-gateway
  namespace: production
  labels:
    app: bitnet-api-gateway
    version: v1
spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 2
  selector:
    matchLabels:
      app: bitnet-api-gateway
  template:
    metadata:
      labels:
        app: bitnet-api-gateway
        version: v1
    spec:
      containers:
      - name: api-gateway
        image: bitnet-rust/api-gateway:1.0.0
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secrets
              key: url
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          allowPrivilegeEscalation: false
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL
```

## CI/CD Pipeline Architecture

### GitHub Actions Workflow
```yaml
name: Production Deployment Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
    
    - name: Run Tests
      run: |
        cargo test --workspace --verbose
        cargo clippy --workspace -- -D warnings
        cargo fmt --all -- --check
    
    - name: Security Audit
      run: |
        cargo install cargo-audit
        cargo audit
    
    - name: Coverage Report
      run: |
        cargo install cargo-tarpaulin
        cargo tarpaulin --workspace --out Xml
    
    - name: Upload Coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and Push Image
      id: build
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS Credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Deploy to Staging
      run: |
        aws eks update-kubeconfig --name bitnet-staging-cluster
        helm upgrade --install bitnet-api ./helm/bitnet-api \
          --namespace staging \
          --set image.tag=${{ github.sha }} \
          --set environment=staging

  deploy-production:
    needs: [build, deploy-staging]
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS Credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Deploy to Production
      run: |
        aws eks update-kubeconfig --name bitnet-prod-cluster
        helm upgrade --install bitnet-api ./helm/bitnet-api \
          --namespace production \
          --set image.tag=${{ github.sha }} \
          --set environment=production \
          --wait --timeout=600s
```

## Monitoring & Observability Stack

### Prometheus Configuration
```yaml
# Prometheus configuration for BitNet monitoring
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "bitnet_alerts.yml"
  - "infrastructure_alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'bitnet-api-gateway'
    kubernetes_sd_configs:
    - role: endpoints
    relabel_configs:
    - source_labels: [__meta_kubernetes_service_name]
      action: keep
      regex: bitnet-api-gateway
    
  - job_name: 'bitnet-quantizer'
    kubernetes_sd_configs:
    - role: endpoints
    relabel_configs:
    - source_labels: [__meta_kubernetes_service_name]
      action: keep
      regex: bitnet-quantizer
    
  - job_name: 'bitnet-inference'
    kubernetes_sd_configs:
    - role: endpoints
    relabel_configs:
    - source_labels: [__meta_kubernetes_service_name]
      action: keep
      regex: bitnet-inference
```

### Critical Alert Rules
```yaml
groups:
- name: bitnet.rules
  rules:
  - alert: HighAPILatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.2
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: High API latency detected
      description: "95th percentile latency is {{ $value }}s"
  
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected
      description: "Error rate is {{ $value }} req/s"
  
  - alert: DatabaseConnectionHigh
    expr: pg_stat_activity_count > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High database connection count
      description: "Database connections: {{ $value }}"
  
  - alert: GPUUtilizationHigh
    expr: nvidia_gpu_utilization > 90
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: High GPU utilization
      description: "GPU utilization: {{ $value }}%"
```

### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "BitNet Production Overview",
    "panels": [
      {
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time p95",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95 Latency"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx Errors/sec"
          }
        ]
      },
      {
        "title": "Active Users",
        "type": "stat",
        "targets": [
          {
            "expr": "bitnet_active_users",
            "legendFormat": "Active Users"
          }
        ]
      }
    ]
  }
}
```

## Security & Compliance Framework

### Network Security Configuration
```yaml
# VPC and Security Group configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: network-config
data:
  vpc-cidr: "10.0.0.0/16"
  private-subnets: "10.0.1.0/24,10.0.2.0/24,10.0.3.0/24"
  public-subnets: "10.0.101.0/24,10.0.102.0/24,10.0.103.0/24"
  
---
# Network Policies for micro-segmentation
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: bitnet-api-policy
spec:
  podSelector:
    matchLabels:
      app: bitnet-api-gateway
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: bitnet-quantizer
    ports:
    - protocol: TCP
      port: 8081
  - to: []
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
```

### Secret Management
```yaml
# External Secrets Operator configuration
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-credentials
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: database-secrets
    creationPolicy: Owner
  data:
  - secretKey: url
    remoteRef:
      key: bitnet/database
      property: connection_string
```

## Disaster Recovery & Business Continuity

### Backup Strategy
```yaml
# Database backup configuration
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:15
            command:
            - /bin/bash
            - -c
            - |
              pg_dump $DATABASE_URL | gzip > /backup/bitnet-$(date +%Y%m%d).sql.gz
              aws s3 cp /backup/bitnet-$(date +%Y%m%d).sql.gz s3://bitnet-backups/database/
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: database-secrets
                  key: url
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          restartPolicy: OnFailure
          volumes:
          - name: backup-storage
            emptyDir: {}
```

### Disaster Recovery Procedures
1. **Automated Failover**: RDS Multi-AZ for database, ALB health checks for applications
2. **Cross-Region Replication**: S3 cross-region replication, RDS read replicas
3. **Recovery Testing**: Monthly DR drills with recovery time validation
4. **Documentation**: Runbooks for common failure scenarios and escalation procedures

## Performance Optimization & Capacity Planning

### Resource Optimization
- **Horizontal Pod Autoscaler**: CPU/memory-based scaling with custom metrics
- **Vertical Pod Autoscaler**: Right-sizing for optimal resource utilization
- **Cluster Autoscaler**: Node-level scaling based on pod resource requirements
- **GPU Resource Management**: NVIDIA GPU operator for efficient GPU utilization

### Capacity Planning Metrics
```yaml
Capacity Planning KPIs:
  CPU Utilization: Target 70% average, 90% peak
  Memory Utilization: Target 75% average, 85% peak  
  GPU Utilization: Target 80% average, 95% peak
  Network Bandwidth: Monitor ingress/egress patterns
  Storage Growth: Track data growth and retention policies
  
Scaling Triggers:
  - CPU > 70% for 5 minutes: Scale out
  - Memory > 75% for 5 minutes: Scale out  
  - Queue depth > 100: Add workers
  - Response time > 500ms p95: Investigate and scale
```

## Operational Procedures

### Incident Response Process
1. **Detection**: Automated alerts via PagerDuty, Slack notifications
2. **Assessment**: Initial triage and severity classification
3. **Response**: Follow incident response runbooks and escalation matrix
4. **Resolution**: Implement fix, validate, and document
5. **Post-Incident Review**: Root cause analysis and improvement actions

### Change Management Process
1. **Change Request**: RFC process for significant infrastructure changes
2. **Approval**: Technical and business stakeholder approval
3. **Testing**: Staging environment validation and rollback planning
4. **Deployment**: Gradual rollout with monitoring and validation
5. **Validation**: Post-deployment testing and performance verification

## Key Performance Indicators (KPIs)

### Infrastructure KPIs
```yaml
Availability Metrics:
  - System uptime: >99.9% (Target: 99.95%)
  - Mean Time to Recovery (MTTR): <15 minutes
  - Mean Time Between Failures (MTBF): >720 hours
  
Performance Metrics:
  - API response time p95: <100ms
  - Database query time p95: <50ms
  - GPU utilization efficiency: >80%
  
Security Metrics:
  - Security vulnerabilities: 0 critical, <5 high
  - Patch compliance: >95% within 30 days
  - Failed authentication attempts: Monitor and alert
  
Cost Metrics:
  - Infrastructure cost per customer: Optimize and track
  - Resource utilization efficiency: >70% across all resources
  - Cloud cost optimization: Monthly cost analysis
```

## Implementation Roadmap

### Phase 1: Foundation Infrastructure (Weeks 1-4)
- [ ] AWS infrastructure setup with Terraform
- [ ] Kubernetes cluster deployment and configuration
- [ ] CI/CD pipeline implementation
- [ ] Basic monitoring and alerting setup

### Phase 2: Production Hardening (Weeks 5-8)  
- [ ] Security hardening and compliance implementation
- [ ] Advanced monitoring and observability deployment
- [ ] Disaster recovery procedures and testing
- [ ] Performance optimization and auto-scaling configuration

### Phase 3: Operational Excellence (Weeks 9-12)
- [ ] Advanced alerting and incident response automation
- [ ] Capacity planning and cost optimization
- [ ] Multi-region deployment and global load balancing
- [ ] Advanced security monitoring and threat detection

---

## Success Criteria

### Immediate Goals (Weeks 1-4)
- [ ] Production infrastructure deployed with 99.9% availability
- [ ] CI/CD pipeline functional with automated deployments
- [ ] Monitoring and alerting operational with <5 minute detection time
- [ ] Security baseline implemented with vulnerability scanning

### Medium-term Goals (Weeks 5-12)
- [ ] Zero critical vulnerabilities and <5 high severity issues
- [ ] <15 minute MTTR for critical incidents
- [ ] <70% average resource utilization across all components
- [ ] SOC 2 audit readiness with comprehensive compliance controls

### Long-term Vision (6+ Months)
- [ ] Global multi-region deployment with <100ms global latency
- [ ] Fully automated operations with minimal human intervention
- [ ] Industry-leading infrastructure metrics and operational excellence
- [ ] Cost-optimized infrastructure supporting profitable unit economics

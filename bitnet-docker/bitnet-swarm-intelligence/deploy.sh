#!/bin/bash

# BitNet Docker Swarm Intelligence Deployment Script
# Automates the complete deployment and setup process

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="bitnet-swarm"
IMAGE_NAME="bitnet-swarm-intelligence"
VERSION="latest"
API_PORT="8080"
MCP_PORT="8081"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${CYAN}$1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "🔍 Checking Prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is available"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose is available"
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    print_success "Docker daemon is running"
    
    # Check available disk space (require at least 5GB)
    available_space=$(df . | tail -1 | awk '{print $4}')
    required_space=5242880  # 5GB in KB
    if [ "$available_space" -lt "$required_space" ]; then
        print_warning "Low disk space. At least 5GB recommended for models and images."
    else
        print_success "Sufficient disk space available"
    fi
}

# Function to setup directories
setup_directories() {
    print_header "📁 Setting up directories..."
    
    mkdir -p models
    mkdir -p logs
    mkdir -p monitoring
    
    # Ensure agent-config directory exists
    if [ ! -d "agent-config" ]; then
        print_warning "agent-config directory not found. Creating minimal structure..."
        mkdir -p agent-config
        echo "# BitNet Agent Configuration Directory" > agent-config/README.md
    fi
    
    print_success "Directories created"
}

# Function to build the Docker image
build_image() {
    print_header "🔨 Building Docker image..."
    
    if [ "$1" == "--rebuild" ] || [ "$1" == "-r" ]; then
        print_status "Force rebuilding image..."
        docker-compose build --no-cache
    elif docker image inspect "$IMAGE_NAME:$VERSION" &> /dev/null; then
        print_status "Image already exists. Use --rebuild to force rebuild."
    else
        print_status "Building image for the first time..."
        docker-compose build
    fi
    
    print_success "Docker image built successfully"
}

# Function to start the services
start_services() {
    print_header "🚀 Starting BitNet Swarm Intelligence services..."
    
    # Stop any existing containers
    docker-compose down 2>/dev/null || true
    
    # Start services
    docker-compose up -d
    
    print_success "Services started"
}

# Function to wait for services to be ready
wait_for_services() {
    print_header "⏳ Waiting for services to be ready..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
            print_success "API server is ready"
            break
        else
            if [ $attempt -eq $max_attempts ]; then
                print_error "API server failed to start within timeout"
                print_status "Checking logs..."
                docker-compose logs bitnet-swarm
                exit 1
            fi
            print_status "Waiting for API server... (attempt $attempt/$max_attempts)"
            sleep 2
            ((attempt++))
        fi
    done
}

# Function to test the deployment
test_deployment() {
    print_header "🧪 Testing deployment..."
    
    # Test health endpoint
    print_status "Testing health endpoint..."
    health_response=$(curl -s http://localhost:$API_PORT/health)
    if echo "$health_response" | grep -q "healthy"; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
        echo "Response: $health_response"
        exit 1
    fi
    
    # Test agent discovery
    print_status "Testing agent discovery..."
    agents_response=$(curl -s http://localhost:$API_PORT/agents/discover)
    agent_count=$(echo "$agents_response" | jq '. | length' 2>/dev/null || echo "0")
    if [ "$agent_count" -gt "0" ]; then
        print_success "Agent discovery working ($agent_count agents found)"
    else
        print_warning "No agents discovered"
    fi
    
    # Test universal API
    print_status "Testing universal API..."
    api_test_request='{"prompt": "test system status", "context": "deployment test"}'
    api_response=$(curl -s -X POST http://localhost:$API_PORT/api \
        -H "Content-Type: application/json" \
        -d "$api_test_request")
    
    if echo "$api_response" | grep -q "operation_type"; then
        print_success "Universal API working"
    else
        print_warning "Universal API test inconclusive"
    fi
}

# Function to show deployment status
show_status() {
    print_header "📊 Deployment Status"
    
    echo ""
    echo "🐳 Container Status:"
    docker-compose ps
    
    echo ""
    echo "📡 Service Endpoints:"
    echo "  • HTTP API:        http://localhost:$API_PORT"
    echo "  • Health Check:    http://localhost:$API_PORT/health"
    echo "  • Agent Discovery: http://localhost:$API_PORT/agents/discover"
    echo "  • Universal API:   http://localhost:$API_PORT/api"
    echo "  • MCP Server:      localhost:$MCP_PORT"
    
    echo ""
    echo "📁 Directory Structure:"
    echo "  • Agent Configs:   ./agent-config/"
    echo "  • Models:          ./models/"
    echo "  • Logs:            ./logs/"
    
    echo ""
    echo "🔧 Useful Commands:"
    echo "  • View logs:       docker-compose logs -f"
    echo "  • Stop services:   docker-compose down"
    echo "  • Restart:         docker-compose restart"
    echo "  • Shell access:    docker-compose exec bitnet-swarm /bin/bash"
    echo "  • Agent CLI:       docker-compose exec bitnet-swarm /app/bin/agent-config-cli status"
    
    # Show agent framework status if available
    if docker-compose exec -T bitnet-swarm /app/bin/agent-config-cli status 2>/dev/null; then
        echo ""
        echo "🤖 Agent Framework Status:"
        docker-compose exec -T bitnet-swarm /app/bin/agent-config-cli status --verbose
    fi
}

# Function to show example API usage
show_examples() {
    print_header "💡 API Usage Examples"
    
    echo ""
    echo "🔹 Health Check:"
    echo "curl http://localhost:$API_PORT/health"
    
    echo ""
    echo "🔹 Agent Discovery:"
    echo "curl http://localhost:$API_PORT/agents/discover"
    
    echo ""
    echo "🔹 Code Generation:"
    echo 'curl -X POST http://localhost:'"$API_PORT"'/api \'
    echo '  -H "Content-Type: application/json" \'
    echo '  -d '\''{"prompt": "generate a Rust function for BitNet inference"}'\'''
    
    echo ""
    echo "🔹 Code Analysis:"
    echo 'curl -X POST http://localhost:'"$API_PORT"'/api \'
    echo '  -H "Content-Type: application/json" \'
    echo '  -d '\''{"prompt": "analyze this code for performance issues", "content": "fn main() { println!(\"hello\"); }"}'\'''
    
    echo ""
    echo "🔹 System Monitoring:"
    echo 'curl -X POST http://localhost:'"$API_PORT"'/api \'
    echo '  -H "Content-Type: application/json" \'
    echo '  -d '\''{"prompt": "show system status and agent health"}'\'''
    
    echo ""
    echo "🔹 Project Scaffolding:"
    echo 'curl -X POST http://localhost:'"$API_PORT"'/api \'
    echo '  -H "Content-Type: application/json" \'
    echo '  -d '\''{"prompt": "scaffold a new Rust project with BitNet integration"}'\'''
}

# Function to cleanup deployment
cleanup() {
    print_header "🧹 Cleaning up deployment..."
    
    print_status "Stopping services..."
    docker-compose down
    
    if [ "$1" == "--full" ]; then
        print_status "Removing images..."
        docker image rm "$IMAGE_NAME:$VERSION" 2>/dev/null || true
        
        print_status "Removing volumes..."
        docker volume prune -f
        
        print_status "Removing networks..."
        docker network prune -f
    fi
    
    print_success "Cleanup completed"
}

# Function to show help
show_help() {
    echo "BitNet Docker Swarm Intelligence Deployment Script"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  deploy              Deploy the complete system (default)"
    echo "  build               Build Docker images only"
    echo "  start               Start services (assumes images exist)"
    echo "  stop                Stop all services"
    echo "  restart             Restart all services"
    echo "  status              Show deployment status"
    echo "  test                Test the deployment"
    echo "  examples            Show API usage examples"
    echo "  logs                Show service logs"
    echo "  cleanup             Stop services and cleanup"
    echo "  help                Show this help message"
    echo ""
    echo "Options:"
    echo "  --rebuild, -r       Force rebuild of Docker images"
    echo "  --full              Full cleanup (removes images and volumes)"
    echo "  --monitor, -m       Monitor deployment after starting"
    echo ""
    echo "Examples:"
    echo "  $0 deploy           # Deploy the complete system"
    echo "  $0 deploy --rebuild # Deploy with forced image rebuild"
    echo "  $0 status           # Show current deployment status"
    echo "  $0 cleanup --full   # Full cleanup including images"
}

# Main deployment function
deploy() {
    print_header "🎯 BitNet Docker Swarm Intelligence Deployment"
    print_header "=============================================="
    
    check_prerequisites
    setup_directories
    build_image "$@"
    start_services
    wait_for_services
    test_deployment
    show_status
    
    print_success "Deployment completed successfully!"
    
    if [[ "$*" == *"--monitor"* ]] || [[ "$*" == *"-m"* ]]; then
        print_status "Monitoring mode enabled. Press Ctrl+C to exit."
        docker-compose logs -f
    fi
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        shift
        deploy "$@"
        ;;
    "build")
        shift
        check_prerequisites
        setup_directories
        build_image "$@"
        ;;
    "start")
        shift
        start_services
        wait_for_services
        show_status
        ;;
    "stop")
        print_status "Stopping services..."
        docker-compose down
        print_success "Services stopped"
        ;;
    "restart")
        print_status "Restarting services..."
        docker-compose restart
        wait_for_services
        show_status
        ;;
    "status")
        show_status
        ;;
    "test")
        test_deployment
        ;;
    "examples")
        show_examples
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "cleanup")
        shift
        cleanup "$@"
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
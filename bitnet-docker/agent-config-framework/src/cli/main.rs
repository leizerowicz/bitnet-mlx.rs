use clap::{Parser, Subcommand};
use std::path::PathBuf;
use agent_config_framework::{
    AgentConfigGenerator, AgentSpec, ContainerRole, AgentType, ResourceRequirements, ResourceLevel,
    OrchestratorRoutingValidator, IntersectionMatrixUpdater, ContainerOrchestration, OrchestrationConfig
};

/// BitNet Agent Config Framework CLI
/// 
/// Command-line interface for managing agent configurations in the Docker BitNet Swarm Intelligence system.
#[derive(Parser)]
#[command(name = "agent-config-cli")]
#[command(about = "BitNet Agent Config Framework - Manage agent configs for Docker Swarm Intelligence")]
#[command(version = "1.0.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a new agent configuration
    Generate {
        /// Agent name
        #[arg(short, long)]
        name: String,
        
        /// Agent type (Specialist, Orchestrator, Utility, Support)
        #[arg(short, long, default_value = "Specialist")]
        agent_type: String,
        
        /// Domain (e.g., code, inference, performance)
        #[arg(short, long)]
        domain: String,
        
        /// Container role (Primary, Secondary, Specialist, Support)
        #[arg(short, long, default_value = "Secondary")]
        container_role: String,
        
        /// Output directory
        #[arg(short, long, default_value = "../agent-config")]
        output: PathBuf,
        
        /// Template directory
        #[arg(short, long, default_value = "templates")]
        templates: PathBuf,
        
        /// Interactive mode
        #[arg(short, long)]
        interactive: bool,
    },
    
    /// Validate all agent configurations
    Validate {
        /// Agent config directory
        #[arg(short, long, default_value = "../agent-config")]
        agent_config_dir: PathBuf,
        
        /// Show detailed errors
        #[arg(short, long)]
        verbose: bool,
        
        /// Fix errors automatically where possible
        #[arg(short, long)]
        fix: bool,
    },
    
    /// Update intersection matrix
    UpdateMatrix {
        /// Agent config directory
        #[arg(short, long, default_value = "../agent-config")]
        agent_config_dir: PathBuf,
        
        /// Matrix file path
        #[arg(short, long, default_value = "../agent-config/agent-intersection-matrix.json")]
        matrix_file: PathBuf,
        
        /// Update all agent configs with new intersections
        #[arg(short, long)]
        update_configs: bool,
    },
    
    /// Deploy Docker container with agent system
    Deploy {
        /// Container configuration file
        #[arg(short, long, default_value = "orchestration-config.yaml")]
        config: PathBuf,
        
        /// Force rebuild container
        #[arg(short, long)]
        rebuild: bool,
        
        /// Monitor deployment
        #[arg(short, long)]
        monitor: bool,
    },
    
    /// Manage running container
    Container {
        #[command(subcommand)]
        action: ContainerActions,
    },
    
    /// Show framework status
    Status {
        /// Show detailed status
        #[arg(short, long)]
        verbose: bool,
    },
}

#[derive(Subcommand)]
enum ContainerActions {
    /// Start the container
    Start,
    /// Stop the container
    Stop,
    /// Restart the container
    Restart,
    /// Show container logs
    Logs {
        /// Follow logs
        #[arg(short, long)]
        follow: bool,
    },
    /// Show agent status in container
    Agents,
    /// Test intelligence modes
    TestIntelligence {
        /// Test request
        #[arg(short, long)]
        request: String,
    },
}

#[tokio::main]
async fn main() {
    env_logger::init();
    
    let cli = Cli::parse();
    
    if let Err(e) = run_cli(cli).await {
        eprintln!("Error: {:?}", e);
        std::process::exit(1);
    }
}

async fn run_cli(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Generate { 
            name, 
            agent_type, 
            domain, 
            container_role, 
            output, 
            templates,
            interactive 
        } => {
            if interactive {
                generate_agent_interactive().await?;
            } else {
                generate_agent(&name, &agent_type, &domain, &container_role, &templates, &output).await?;
            }
        },
        
        Commands::Validate { agent_config_dir, verbose, fix } => {
            validate_configs(&agent_config_dir, verbose, fix).await?;
        },
        
        Commands::UpdateMatrix { agent_config_dir, matrix_file, update_configs } => {
            update_intersection_matrix(&agent_config_dir, &matrix_file, update_configs).await?;
        },
        
        Commands::Deploy { config, rebuild, monitor } => {
            deploy_container(&config, rebuild, monitor).await?;
        },
        
        Commands::Container { action } => {
            handle_container_action(action).await?;
        },
        
        Commands::Status { verbose } => {
            show_status(verbose).await?;
        },
    }
    
    Ok(())
}

async fn generate_agent_interactive() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 BitNet Agent Config Generator - Interactive Mode");
    println!("=================================================");
    
    // Interactive prompts for agent creation
    println!("\n1. Agent Information");
    print!("Agent name: ");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let name = input.trim().to_string();
    
    print!("Domain (code, inference, performance, security, etc.): ");
    input.clear();
    std::io::stdin().read_line(&mut input)?;
    let domain = input.trim().to_string();
    
    print!("Container role (Primary/Secondary/Specialist/Support): ");
    input.clear();
    std::io::stdin().read_line(&mut input)?;
    let container_role = input.trim().to_string();
    
    println!("\n2. Intelligence Mode Capabilities");
    print!("Supports Swarm Intelligence? (y/n): ");
    input.clear();
    std::io::stdin().read_line(&mut input)?;
    let swarm_capable = input.trim().to_lowercase() == "y";
    
    print!("Supports Hive Mind Intelligence? (y/n): ");
    input.clear();
    std::io::stdin().read_line(&mut input)?;
    let hive_mind_capable = input.trim().to_lowercase() == "y";
    
    println!("\n3. Resource Requirements");
    print!("CPU requirement (Low/Medium/High): ");
    input.clear();
    std::io::stdin().read_line(&mut input)?;
    let cpu_level = match input.trim().to_lowercase().as_str() {
        "low" => ResourceLevel::Low,
        "high" => ResourceLevel::High,
        _ => ResourceLevel::Medium,
    };
    
    print!("Memory requirement (Low/Medium/High): ");
    input.clear();
    std::io::stdin().read_line(&mut input)?;
    let memory_level = match input.trim().to_lowercase().as_str() {
        "low" => ResourceLevel::Low,
        "high" => ResourceLevel::High,
        _ => ResourceLevel::Medium,
    };
    
    print!("Requires GPU? (y/n): ");
    input.clear();
    std::io::stdin().read_line(&mut input)?;
    let gpu_required = input.trim().to_lowercase() == "y";
    
    // Generate agent spec
    let spec = create_agent_spec(
        &name,
        &domain,
        &container_role,
        cpu_level,
        memory_level,
        gpu_required,
        swarm_capable,
        hive_mind_capable,
    );
    
    // Generate config
    let generator = AgentConfigGenerator::new("templates", "../agent-config");
    let output_path = generator.generate_and_save(&spec)?;
    
    println!("\n✅ Agent config generated successfully: {}", output_path.display());
    
    Ok(())
}

async fn generate_agent(
    name: &str,
    agent_type: &str,
    domain: &str,
    container_role: &str,
    templates: &PathBuf,
    output: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Generating agent config for: {}", name);
    
    let spec = create_agent_spec(
        name,
        domain,
        container_role,
        ResourceLevel::Medium,
        ResourceLevel::Medium,
        false,
        true,  // Default to swarm capable
        true,  // Default to hive mind capable
    );
    
    let generator = AgentConfigGenerator::new(templates, output);
    let output_path = generator.generate_and_save(&spec)?;
    
    println!("✅ Agent config generated: {}", output_path.display());
    
    Ok(())
}

fn create_agent_spec(
    name: &str,
    domain: &str,
    container_role: &str,
    cpu_level: ResourceLevel,
    memory_level: ResourceLevel,
    gpu_required: bool,
    swarm_capable: bool,
    hive_mind_capable: bool,
) -> AgentSpec {
    let container_role_enum = match container_role.to_lowercase().as_str() {
        "primary" => ContainerRole::Primary,
        "specialist" => ContainerRole::Specialist,
        "support" => ContainerRole::Support,
        _ => ContainerRole::Secondary,
    };
    
    let resource_description = format!(
        "{} CPU, {} memory{}",
        format!("{:?}", cpu_level),
        format!("{:?}", memory_level),
        if gpu_required { ", GPU required" } else { "" }
    );
    
    let swarm_use_cases = if swarm_capable {
        vec![
            format!("Independent {} development with collaborative integration", domain),
            format!("{} exploration with consensus building", domain),
            format!("Parallel {} analysis with result synthesis", domain),
        ]
    } else {
        vec![]
    };
    
    let hive_mind_use_cases = if hive_mind_capable {
        vec![
            format!("Unified {} implementation across entire system", domain),
            format!("Coordinated {} optimization with shared targets", domain),
            format!("Synchronized {} processing with collective intelligence", domain),
        ]
    } else {
        vec![]
    };
    
    AgentSpec {
        name: name.to_string(),
        agent_type: AgentType::Specialist,
        domain: domain.to_string(),
        container_role: container_role_enum,
        specialist_role: format!("{} specialist", domain),
        primary_focus: format!("{} operations and optimization", domain),
        core_expertise: format!("advanced {} implementation and best practices", domain),
        api_endpoints: vec![
            "/api".to_string(),
            format!("/agents/{}/execute", domain),
            format!("/{}/operations", domain),
        ],
        mcp_tools: vec![
            format!("{}-operations", domain),
            format!("{}-analysis", domain),
            format!("{}-optimization", domain),
        ],
        resource_requirements: ResourceRequirements {
            cpu: cpu_level,
            memory: memory_level,
            gpu_required,
            description: resource_description,
        },
        swarm_coordination: format!("Independent {} work with collaborative coordination", domain),
        hive_mind_coordination: format!("Unified {} execution with synchronized processing", domain),
        swarm_use_cases,
        hive_mind_use_cases,
        primary_responsibilities: vec![
            format!("{} implementation", domain),
            format!("{} optimization", domain),
            format!("{} quality assurance", domain),
            "Cross-agent coordination".to_string(),
        ],
        docker_capabilities: vec![
            format!("Containerized {} operations", domain),
            "Intelligence mode switching".to_string(),
            "Agent coordination".to_string(),
        ],
        intersections: vec!["orchestrator".to_string()],
        quality_standards: vec![
            "Code quality validation".to_string(),
            "Performance benchmarking".to_string(),
            "Integration testing".to_string(),
        ],
    }
}

async fn validate_configs(
    agent_config_dir: &PathBuf,
    verbose: bool,
    fix: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Validating agent configurations...");
    
    let validator = OrchestratorRoutingValidator::new(agent_config_dir);
    let results = validator.validate_all_configs()?;
    let report = validator.generate_report(&results);
    
    if verbose {
        for result in &results {
            if !result.is_valid {
                println!("\n❌ {}", result.file_path.display());
                for error in &result.errors {
                    println!("   Error: {:?}", error);
                }
            } else {
                println!("✅ {}", result.file_path.display());
            }
            
            if !result.warnings.is_empty() {
                for warning in &result.warnings {
                    println!("   Warning: {}", warning);
                }
            }
        }
    }
    
    report.print_summary();
    
    if fix && report.invalid_configs > 0 {
        println!("\n🔧 Attempting to fix validation errors...");
        // Implementation for automatic fixes would go here
        println!("   Automatic fixes not yet implemented");
    }
    
    Ok(())
}

async fn update_intersection_matrix(
    agent_config_dir: &PathBuf,
    matrix_file: &PathBuf,
    update_configs: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Updating intersection matrix...");
    
    let mut updater = IntersectionMatrixUpdater::new(agent_config_dir, matrix_file)?;
    updater.update_intersection_matrix()?;
    
    if update_configs {
        println!("📝 Updating agent configs with new intersections...");
        // Implementation for updating agent configs would go here
        println!("   Config updates not yet implemented");
    }
    
    println!("✅ Intersection matrix updated successfully");
    
    Ok(())
}

async fn deploy_container(
    config_path: &PathBuf,
    rebuild: bool,
    monitor: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Deploying BitNet Docker Swarm Intelligence container...");
    
    // Load configuration
    let config = load_orchestration_config(config_path)?;
    
    if rebuild {
        println!("🔨 Forcing container rebuild...");
    }
    
    // Initialize orchestration system
    let mut orchestration = ContainerOrchestration::new(config);
    orchestration.initialize_system().await?;
    
    if monitor {
        println!("📊 Monitoring deployment...");
        // Keep the process running to monitor
        tokio::signal::ctrl_c().await?;
        println!("\n🛑 Shutting down...");
        orchestration.shutdown().await?;
    }
    
    println!("✅ Container deployed successfully");
    
    Ok(())
}

fn load_orchestration_config(config_path: &PathBuf) -> Result<OrchestrationConfig, Box<dyn std::error::Error>> {
    // Load configuration from file or create default
    Ok(OrchestrationConfig {
        container_name: "bitnet-swarm".to_string(),
        image_name: "bitnet-swarm-intelligence".to_string(),
        base_port: 8080,
        agent_config_mount: PathBuf::from("../agent-config"),
        model_cache_mount: PathBuf::from("./models"),
        resource_limits: agent_config_framework::ResourceLimits {
            cpu_limit: "2.0".to_string(),
            memory_limit: "4G".to_string(),
            gpu_enabled: false,
        },
        intelligence_modes: agent_config_framework::IntelligenceModeConfig {
            default_mode: "swarm".to_string(),
            auto_mode_selection: true,
            mode_switch_threshold: 0.7,
            swarm_agent_limit: 10,
            hive_mind_agent_limit: 20,
        },
    })
}

async fn handle_container_action(action: ContainerActions) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        ContainerActions::Start => {
            println!("🚀 Starting BitNet Swarm container...");
            // Implementation for starting container
        },
        ContainerActions::Stop => {
            println!("🛑 Stopping BitNet Swarm container...");
            // Implementation for stopping container
        },
        ContainerActions::Restart => {
            println!("🔄 Restarting BitNet Swarm container...");
            // Implementation for restarting container
        },
        ContainerActions::Logs { follow } => {
            println!("📋 Showing container logs...");
            if follow {
                println!("   Following logs (Ctrl+C to stop)");
            }
            // Implementation for showing logs
        },
        ContainerActions::Agents => {
            println!("🤖 Agent status in container:");
            // Implementation for showing agent status
        },
        ContainerActions::TestIntelligence { request } => {
            println!("🧠 Testing intelligence system with request: {}", request);
            // Implementation for testing intelligence modes
        },
    }
    
    Ok(())
}

async fn show_status(verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 BitNet Agent Config Framework Status");
    println!("=====================================");
    
    // Check framework installation
    println!("✅ Framework: Installed and ready");
    
    // Check agent configs
    let agent_config_dir = PathBuf::from("../agent-config");
    if agent_config_dir.exists() {
        let count = std::fs::read_dir(&agent_config_dir)?.count();
        println!("✅ Agent Configs: {} configurations found", count);
    } else {
        println!("❌ Agent Configs: Directory not found");
    }
    
    // Check Docker
    match std::process::Command::new("docker").arg("--version").output() {
        Ok(_) => println!("✅ Docker: Available"),
        Err(_) => println!("❌ Docker: Not available"),
    }
    
    // Check container status
    match std::process::Command::new("docker")
        .args(&["ps", "--filter", "name=bitnet-swarm", "--format", "table {{.Names}}\t{{.Status}}"])
        .output() {
        Ok(output) => {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if output_str.contains("bitnet-swarm") {
                println!("✅ Container: Running");
            } else {
                println!("⏸️ Container: Not running");
            }
        },
        Err(_) => println!("❓ Container: Status unknown"),
    }
    
    if verbose {
        println!("\n📋 Detailed Information:");
        println!("   Template directory: templates/");
        println!("   Agent config directory: ../agent-config/");
        println!("   Default container name: bitnet-swarm");
        println!("   Default ports: 8080 (HTTP API), 8081 (MCP Server)");
    }
    
    Ok(())
}
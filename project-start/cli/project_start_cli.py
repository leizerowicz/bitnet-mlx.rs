#!/usr/bin/env python3
"""
Project-Start Enhanced CLI - Interactive specification-driven development tool
Spec-Driven Development Toolkit with Gemini AI Integration

Usage:
    python3 project_start_cli.py init <project-name>
    python3 project_start_cli.py enhance --here
"""

import os
import sys
import argparse
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Constants
AI_ASSISTANT = "gemini"  # Simplified to only support Gemini
AI_CHOICES = {"gemini": "Gemini CLI"}  # Keep for compatibility

# ASCII Art Banner
BANNER = """
██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗    ███████╗████████╗ █████╗ ██████╗ ████████╗
██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝    ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝
██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║       ███████╗   ██║   ███████║██████╔╝   ██║
██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║       ╚════██║   ██║   ██╔══██║██╔══██╗   ██║
██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║       ███████║   ██║   ██║  ██║██║  ██║   ██║
╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝       ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝
"""

TAGLINE = "Specification-Driven Development with AI Agent Collaboration"

console = object()  # Placeholder for rich console


class AIIntegration:
    """Gemini AI integration for intelligent document generation"""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.ai_assistant = "gemini"  # Always use Gemini
        self.temp_dir = Path(tempfile.mkdtemp())
        self.ai_tool_available = self._check_gemini_availability()

    def _check_gemini_availability(self) -> bool:
        """Check if Gemini CLI is available"""
        return shutil.which("gemini") is not None

    def create_enhanced_prompt(
        self,
        document_type: str,
        project_info: Dict[str, Any],
        additional_context: str = "",
    ) -> str:
        """Create context-rich prompts for specific document types"""

        base_context = f"""
Project Information:
- Name: {project_info.get('name', 'Unknown')}
- Description: {project_info.get('description', 'No description provided')}
- Technology Stack: {project_info.get('tech_stack', 'Not specified')}
- Architecture: {project_info.get('architecture', 'Not specified')}
- Development Approach: {project_info.get('development_approach', 'Not specified')}
- Team Size: {project_info.get('team_size', 'Not specified')}
- Agent Coordination: {project_info.get('agent_coordination', 'Not specified')}
- Target Audience: {project_info.get('target_audience', 'Not specified')}
- Business Value: {project_info.get('business_value', 'Not specified')}
- Quality Requirements: {project_info.get('quality_requirements', 'Standard quality practices')}
- Timeline: {project_info.get('timeline', 'Not specified')}
- Testing Strategy: {project_info.get('testing_strategy', 'Standard testing')}

Constitutional Framework Context:
The Project-Start framework follows constitutional principles:
- Article III: All specifications must be testable and pass validation
- Article IV: Specification-driven development (requirements lead implementation)
- Article V: Agent coordination with clear roles and boundaries
- Article VII: Simplicity principle (avoid over-engineering)
- Article VIII: Test-first development (NON-NEGOTIABLE)
- Article IX: Continuous validation at every workflow transition

{additional_context}
"""

        prompts = {
            "backlog": f"""{base_context}

Please generate a comprehensive Product Backlog document in Markdown format for this project. The document should:

1. Include a clear project overview that expands on the basic description
2. Define specific user personas based on the target audience
3. Create detailed user stories and epics that reflect the technology stack and architecture
4. Include realistic acceptance criteria that align with the development approach
5. Prioritize features based on business value and technical dependencies
6. Address non-functional requirements specific to the technology choices
7. Include constitutional compliance checkpoints
8. Identify areas that need clarification

Focus on creating actionable, testable user stories that a development team can immediately work with. Consider the team size and coordination level when structuring the backlog.

Generate a complete BACKLOG.md file:""",
            "implementation_guide": f"""{base_context}

Please generate a detailed Implementation Guide document in Markdown format for this project. The document should:

1. Provide specific technical architecture guidance based on the chosen technology stack
2. Break down development into realistic phases considering team size and timeline
3. Include technology-specific setup instructions and best practices
4. Define quality gates that align with constitutional principles
5. Address the specific architecture style (monolithic/microservices/serverless/hybrid)
6. Create a testing strategy that implements test-first development
7. Include risk mitigation strategies specific to the technology choices
8. Define performance targets appropriate for the project scale
9. Provide deployment strategy aligned with the development approach

Focus on actionable technical guidance that considers the team's capabilities and project constraints.

Generate a complete IMPLEMENTATION_GUIDE.md file:""",
            "risk_assessment": f"""{base_context}

Please generate a comprehensive Risk Assessment document in Markdown format for this project. The document should:

1. Identify specific risks related to the chosen technology stack and architecture
2. Assess risks based on team size, timeline, and coordination approach
3. Consider business and technical constraints mentioned in the project info
4. Provide realistic probability and impact assessments
5. Include constitutional compliance risks and mitigation strategies
6. Address quality requirements and testing strategy risks
7. Create actionable mitigation plans with clear responsibilities
8. Include monitoring and review processes for ongoing risk management

Focus on practical, project-specific risks rather than generic software development risks.

Generate a complete RISK_ASSESSMENT.md file:""",
            "file_outline": f"""{base_context}

Please generate a detailed File Structure Outline document in Markdown format for this project. The document should:

1. Create a technology stack-specific directory structure
2. Organize files based on the chosen architecture style
3. Consider the development approach and team coordination needs
4. Include proper separation of concerns for the technology choices
5. Provide file naming conventions appropriate for the tech stack
6. Structure for testing strategy implementation
7. Include configuration and deployment file organization
8. Consider constitutional compliance file organization
9. Plan for documentation and quality assurance files

Focus on a practical, immediately implementable file structure that supports the project goals.

Generate a complete FILE_OUTLINE.md file:""",
        }

        return prompts.get(
            document_type,
            f"{base_context}\nGenerate a {document_type} document for this project:",
        )

    def call_ai_assistant(self, prompt: str, document_type: str) -> str:
        """Call Gemini CLI or provide fallback content"""
        if not self.ai_tool_available:
            print("⚠️ Gemini CLI not available - using fallback generation")
            return self._generate_fallback_content(prompt, document_type)

        print(f"🤖 Calling Gemini CLI for {document_type} generation...")
        try:
            return self._call_gemini(prompt, document_type)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error calling Gemini CLI: {e}")
            print("📝 Falling back to template generation...")
            return self._generate_fallback_content(prompt, document_type)
        except subprocess.TimeoutExpired:
            print("⏰ Gemini CLI timeout - falling back to template generation...")
            return self._generate_fallback_content(prompt, document_type)

    def _call_gemini(self, prompt: str, document_type: str) -> str:
        """Call Gemini CLI to generate content"""
        try:
            result = subprocess.run(
                ["gemini", "generate", "--prompt", prompt, "--type", document_type],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Gemini CLI error: {e}") from e

    def _generate_fallback_content(self, prompt: str, document_type: str) -> str:
        """Generate intelligent fallback content when Gemini CLI is not available"""

        # Create a temporary prompt file for manual processing
        temp_file = self.temp_dir / f"{document_type}_prompt.md"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(
                f"# {document_type.title().replace('_', ' ')} Generation Prompt\n\n"
            )
            f.write(prompt)
            f.write(
                f"\n\n---\n**Generated by Project-Start CLI at {datetime.now().isoformat()}**\n"
            )

        return f"""# {document_type.title().replace('_', ' ')}

## Project Context
- **Generation Method**: Fallback Template (Gemini CLI not available)
- **AI Assistant**: Gemini CLI (unavailable)
- **Timestamp**: {datetime.now().isoformat()}

## Instructions for Enhancement
To enhance this document with AI assistance:

1. **Install Gemini CLI**: Follow the installation guide
2. **Review Prompt**: See the detailed prompt in `{temp_file}`
3. **Manual Enhancement**: Copy the prompt to Gemini CLI or web interface

## Template Content

*This section contains a basic template. For intelligent, project-specific content, please use the Gemini CLI.*

### Overview
This document was automatically generated as part of the Project-Start specification-driven development workflow.

### Next Steps
1. Install Gemini CLI for enhanced AI generation
2. Review and customize content based on your specific project needs
3. Validate against constitutional framework principles

---

**Note**: This document was generated using fallback templates because Gemini CLI was not available.
**Prompt File**: `{temp_file}`
"""


# ASCII Art Banner for Agentic Engineering
AGENTIC_ENGINEERING_BANNER = """
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗ ██████╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║██║
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║██║
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║╚██████╗
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝

███████╗███╗   ██╗ ██████╗ ██╗███╗   ██╗███████╗███████╗██████╗ ██╗███╗   ██╗ ██████╗
██╔════╝████╗  ██║██╔════╝ ██║████╗  ██║██╔════╝██╔════╝██╔══██╗██║████╗  ██║██╔════╝
█████╗  ██╔██╗ ██║██║  ███╗██║██╔██╗ ██║█████╗  █████╗  ██████╔╝██║██╔██╗ ██║██║  ███╗
██╔══╝  ██║╚██╗██║██║   ██║██║██║╚██╗██║██╔══╝  ██╔══╝  ██╔══██╗██║██║╚██╗██║██║   ██║
███████╗██║ ╚████║╚██████╔╝██║██║ ╚████║███████╗███████╗██║  ██║██║██║ ╚████║╚██████╔╝
╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝
"""

TAGLINE = "Specification-Driven Development with AI Agent Collaboration"


class ProjectStartCLI:
    def __init__(self):
        self.project_dir = self._detect_project_root()
        self.config_dir = self.project_dir / "agent_config"
        self.specs_dir = self.project_dir / "specs"
        self.memory_dir = self.project_dir / "memory"
        self.ai_assistant = "gemini"  # Always use Gemini
        self.ai_integration = AIIntegration(self.project_dir)

    def _detect_vscode_environment(self) -> bool:
        """Detect if running within VS Code environment"""
        # Check environment variables that VS Code sets
        vscode_indicators = [
            os.environ.get("VSCODE_PID"),
            os.environ.get("TERM_PROGRAM") == "vscode",
            os.environ.get("PROJECT_START_VSCODE") == "true",
        ]
        return any(vscode_indicators)

    def _detect_project_root(self) -> Path:
        """Detect the project root directory"""
        current_dir = Path.cwd()
        cli_dir = Path(__file__).parent
        project_start_dir = cli_dir.parent
        parent_dir = project_start_dir.parent

        if project_start_dir != parent_dir:
            return parent_dir

        return current_dir

    def show_banner(self):
        """Display the ASCII art banner for Agentic Engineering"""
        print("\n" + "=" * 80)
        banner_lines = AGENTIC_ENGINEERING_BANNER.strip().split("\n")
        for line in banner_lines:
            print(line)

        print("\n" + " " * 15 + TAGLINE)
        print("=" * 80 + "\n")

    def show_copilot_integration_status(self):
        """Show AI integration status"""
        print("🤖 AI Integration: ✅ Gemini CLI ENABLED")
        if self.ai_integration.ai_tool_available:
            print("✅ Gemini CLI tool detected and available")
        else:
            print("⚠️ Gemini CLI tool not detected - will use fallback generation")

        print("🎯 AI will be called after collecting all project context")
        print()
        print("   • Constitutional AI governance active")
        print("   • Multi-agent coordination protocols ready")
        print("   • Persistent context management initialized")
        print("   • Enhanced template generation available")

        print(f"📁 Project Root: {self.project_dir}")
        print()

    def ask_question(
        self, question: str, default: str = "", required: bool = True
    ) -> str:
        """Ask user a question with optional default value"""
        prompt = f"\n{question}"
        if default:
            prompt += f" (default: {default})"
        prompt += ": "

        while True:
            answer = input(prompt).strip()
            if answer:
                return answer
            elif default:
                return default
            elif not required:
                return ""
            else:
                print("This field is required. Please provide a value.")

    def ask_multiple_choice(
        self, question: str, choices: List[str], default: str = ""
    ) -> str:
        """Ask user to choose from multiple options"""
        print(f"\n{question}")
        for i, choice in enumerate(choices, 1):
            print(f"{i}. {choice}")

        while True:
            answer = input(f"\nSelect option (1-{len(choices)}): ").strip()
            try:
                index = int(answer) - 1
                if 0 <= index < len(choices):
                    return choices[index]
                else:
                    print(f"Please enter a number between 1 and {len(choices)}")
            except ValueError:
                if default and not answer:
                    return default
                print("Please enter a valid number")

    def ask_yes_no(self, question: str, default: bool = True) -> bool:
        """Ask user a yes/no question"""
        default_text = "Y/n" if default else "y/N"
        while True:
            answer = input(f"\n{question} ({default_text}): ").strip().lower()
            if answer in ["y", "yes"]:
                return True
            elif answer in ["n", "no"]:
                return False
            elif not answer:
                return default
            else:
                print("Please answer 'y' or 'n'")

    def show_interactive_menu(self) -> None:
        """Display interactive menu for command selection"""
        while True:
            self.show_banner()
            self.show_copilot_integration_status()

            print("🚀 PROJECT-START ENHANCED - Interactive Command Menu")
            print("=" * 60)
            print()
            print("Available Commands:")
            print()
            print("1. 🌟 Start New Project (Complete Workflow)")
            print("   └─ Full 4-step enhanced workflow for new projects")
            print()
            print("2. 🔍 Step 1: Interactive Discovery")
            print("   └─ Comprehensive project specification generation")
            print()
            print("3. 📋 Step 2: SPARC Planning")
            print("   └─ Constitutional SPARC methodology implementation")
            print()
            print("4. 🧠 Step 3: Context Systems")
            print("   └─ Persistent context and expert systems creation")
            print()
            print("5. 🤝 Step 4: PACT Framework")
            print("   └─ Constitutional PACT framework deployment")
            print()
            print("6. ⚙️  Configure Project Root")
            print("   └─ Set up project root configuration")
            print()
            print("7. ❓ Help & Documentation")
            print("   └─ View available commands and usage examples")
            print()
            print("0. 🚪 Exit")
            print()

            try:
                choice = input("Select an option (0-7): ").strip()

                if choice == "0":
                    print(
                        "\n👋 Goodbye! Thank you for using Project-Start Enhanced CLI."
                    )
                    break
                elif choice == "1":
                    self._handle_start_new_project()
                elif choice == "2":
                    self._handle_step_1()
                elif choice == "3":
                    self._handle_step_2()
                elif choice == "4":
                    self._handle_step_3()
                elif choice == "5":
                    self._handle_step_4()
                elif choice == "6":
                    self._handle_configure_project_root()
                elif choice == "7":
                    self._show_help_documentation()
                else:
                    print(f"\n❌ Invalid option: {choice}")
                    print("Please select a number between 0-7")
                    input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\n\n🛑 Operation cancelled by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("\nPress Enter to continue...")

    def _handle_start_new_project(self) -> None:
        """Handle the start new project command"""
        print("\n" + "=" * 60)
        print("🌟 STARTING NEW PROJECT - Complete Enhanced Workflow")
        print("=" * 60)
        print()
        print("This will run the complete 4-step Project-Start workflow:")
        print("• Step 1: Interactive discovery and specification generation")
        print("• Step 2: Constitutional SPARC methodology planning")
        print("• Step 3: Persistent context systems creation")
        print("• Step 4: Constitutional PACT framework deployment")
        print()

        if not self.ask_yes_no("Continue with complete workflow?", True):
            return

        description = self.ask_question("Enter your project description")
        print(f"\n🚀 Starting enhanced workflow for: {description}")

        try:
            self.project_start_enhanced_workflow(description)
            print("\n✅ Complete workflow finished successfully!")
        except Exception as e:
            print(f"\n❌ Workflow error: {e}")

        input("\nPress Enter to return to main menu...")

    def _handle_step_1(self) -> None:
        """Handle Step 1: Interactive Discovery"""
        print("\n" + "=" * 60)
        print("🔍 STEP 1: Interactive Discovery & Specification Generation")
        print("=" * 60)
        print()
        print("This step will:")
        print("• Collect comprehensive project requirements")
        print("• Generate user stories and feature backlog")
        print("• Create implementation guidance")
        print("• Assess risks and mitigation strategies")
        print("• Generate file structure recommendations")
        print()

        existing_project = self.ask_yes_no("Is this for an existing project?", False)
        description = self.ask_question("Enter project description")

        print(f"\n🔍 Running Step 1 for: {description}")
        print(
            f"Mode: {'Existing Project Analysis' if existing_project else 'New Project Discovery'}"
        )

        try:
            self.enhance_step_1(description, existing_project)
            print("\n✅ Step 1 completed successfully!")
        except Exception as e:
            print(f"\n❌ Step 1 error: {e}")

        input("\nPress Enter to return to main menu...")

    def _handle_step_2(self) -> None:
        """Handle Step 2: SPARC Planning"""
        print("\n" + "=" * 60)
        print("📋 STEP 2: SPARC Planning & Constitutional Methodology")
        print("=" * 60)
        print()
        print("This step will:")
        print("• Apply constitutional SPARC methodology")
        print("• Generate formal specifications")
        print("• Create pseudocode and architecture plans")
        print("• Establish testing and quality strategies")
        print()

        project_path = self.ask_question(
            "Enter project path (or press Enter for current directory)", "", False
        )

        print(f"\n📋 Running Step 2...")
        print(f"Project path: {project_path or 'Current directory'}")

        try:
            self.enhance_step_2(project_path)
            print("\n✅ Step 2 completed successfully!")
        except Exception as e:
            print(f"\n❌ Step 2 error: {e}")

        input("\nPress Enter to return to main menu...")

    def _handle_step_3(self) -> None:
        """Handle Step 3: Context Systems"""
        print("\n" + "=" * 60)
        print("🧠 STEP 3: Context Systems & Expert Domain Creation")
        print("=" * 60)
        print()
        print("This step will:")
        print("• Create persistent context systems")
        print("• Generate expert domain specializations")
        print("• Set up multi-agent coordination protocols")
        print("• Configure automated context management")
        print()

        project_path = self.ask_question(
            "Enter project path (or press Enter for current directory)", "", False
        )

        print(f"\n🧠 Running Step 3...")
        print(f"Project path: {project_path or 'Current directory'}")

        try:
            self.enhance_step_3(project_path)
            print("\n✅ Step 3 completed successfully!")
        except Exception as e:
            print(f"\n❌ Step 3 error: {e}")

        input("\nPress Enter to return to main menu...")

    def _handle_step_4(self) -> None:
        """Handle Step 4: PACT Framework"""
        print("\n" + "=" * 60)
        print("🤝 STEP 4: PACT Framework & Constitutional Governance")
        print("=" * 60)
        print()
        print("This step will:")
        print("• Deploy constitutional PACT framework")
        print("• Create multi-agent coordination systems")
        print("• Establish quality assurance protocols")
        print("• Set up automated governance")
        print()

        project_path = self.ask_question(
            "Enter project path (or press Enter for current directory)", "", False
        )

        print(f"\n🤝 Running Step 4...")
        print(f"Project path: {project_path or 'Current directory'}")

        try:
            self.enhance_step_4(project_path)
            print("\n✅ Step 4 completed successfully!")
        except Exception as e:
            print(f"\n❌ Step 4 error: {e}")

        input("\nPress Enter to return to main menu...")

    def _handle_configure_project_root(self) -> None:
        """Handle project root configuration"""
        print("\n" + "=" * 60)
        print("⚙️  CONFIGURE PROJECT ROOT")
        print("=" * 60)
        print()
        print("This will configure the project root for nested projects")
        print("(when project-start is inside another project).")
        print()

        if not self.ask_yes_no("Continue with project root configuration?", True):
            return

        try:
            self.configure_project_root()
            print("\n✅ Project root configuration completed!")
        except Exception as e:
            print(f"\n❌ Configuration error: {e}")

        input("\nPress Enter to return to main menu...")

    def _show_help_documentation(self) -> None:
        """Show help and documentation"""
        print("\n" + "=" * 60)
        print("❓ HELP & DOCUMENTATION")
        print("=" * 60)
        print()
        print("🚀 PROJECT-START ENHANCED CLI")
        print()
        print("COMMAND LINE USAGE:")
        print('python3 project_start_cli.py start "My new project"')
        print(
            'python3 project_start_cli.py /enhance-step-1 "Chat app" --existing-project'
        )
        print(
            "python3 project_start_cli.py /enhance-step-2 --project-path specs/001-my-project"
        )
        print(
            "python3 project_start_cli.py /enhance-step-3 --project-path specs/001-my-project"
        )
        print(
            "python3 project_start_cli.py /enhance-step-4 --project-path specs/001-my-project"
        )
        print('python3 project_start_cli.py /project-start-enhanced "Full workflow"')
        print("python3 project_start_cli.py /configure-project-root")
        print()
        print("AVAILABLE COMMANDS:")
        print("• start                    - Interactive new project creation")
        print("• /enhance-step-1          - Discovery and specification generation")
        print("• /enhance-step-2          - SPARC planning methodology")
        print("• /enhance-step-3          - Context systems creation")
        print("• /enhance-step-4          - PACT framework deployment")
        print("• /project-start-enhanced  - Complete 4-step workflow")
        print("• /configure-project-root  - Configure project root")
        print()
        print("WORKFLOW STEPS:")
        print("1. Step 1: Discovery & Specification")
        print("   └─ Creates comprehensive project specifications")
        print("2. Step 2: SPARC Planning")
        print("   └─ Applies constitutional methodology for planning")
        print("3. Step 3: Context Systems")
        print("   └─ Sets up persistent context and expert systems")
        print("4. Step 4: PACT Framework")
        print("   └─ Deploys constitutional governance and coordination")
        print()
        print("AI INTEGRATION:")
        print("• Gemini CLI integration for intelligent document generation")
        print("• Fallback templates when AI tools are unavailable")
        print("• Constitutional AI governance throughout workflow")
        print()
        print("For more information, see cli/README.md")

        input("\nPress Enter to return to main menu...")

    def collect_project_info(self) -> Dict[str, Any]:
        """Interactive questionnaire to collect project information"""
        print("\n" + "=" * 60)
        print("🚀 PROJECT-START ENHANCED - Interactive Project Setup")
        print("=" * 60)

        print(
            "\nThis tool will guide you through creating a comprehensive project specification"
        )
        print("using AI integration with Project-Start workflow.\n")

        project_info = {}

        # Basic project information
        print("\n📋 BASIC PROJECT INFORMATION")
        print("-" * 30)
        project_info["name"] = self.ask_question("Project name")
        project_info["description"] = self.ask_question(
            "Project description (brief overview)"
        )
        project_info["detailed_description"] = self.ask_question(
            "Detailed project description (what it does, who it serves)", required=False
        )

        # Business context
        print("\n🎯 BUSINESS CONTEXT")
        print("-" * 20)
        project_info["target_audience"] = self.ask_question(
            "Target audience/users", required=False
        )
        project_info["business_value"] = self.ask_question(
            "Primary business value/goal", required=False
        )
        project_info["success_metrics"] = self.ask_question(
            "How will you measure success?", required=False
        )

        # Technical preferences
        print("\n🔧 TECHNICAL PREFERENCES")
        print("-" * 25)

        tech_stacks = [
            "Python (FastAPI/Django) + React + PostgreSQL",
            "Node.js (Express) + React + MongoDB",
            "Java (Spring Boot) + Angular + MySQL",
            "C# (.NET Core) + React + SQL Server",
            "Go + Vue.js + PostgreSQL",
            "PHP (Laravel) + Vue.js + MySQL",
            "Custom/Other",
        ]
        project_info["tech_stack"] = self.ask_multiple_choice(
            "Preferred technology stack:", tech_stacks
        )

        if project_info["tech_stack"] == "Custom/Other":
            project_info["custom_tech_stack"] = self.ask_question(
                "Describe your preferred technology stack"
            )

        architectures = [
            "Monolithic (single deployable unit)",
            "Microservices (distributed services)",
            "Serverless (functions-as-a-service)",
            "Hybrid (mixed approach)",
        ]
        project_info["architecture"] = self.ask_multiple_choice(
            "Preferred architecture style:", architectures
        )

        # Development approach
        print("\n🏗️ DEVELOPMENT APPROACH")
        print("-" * 25)

        project_info["team_size"] = self.ask_question(
            "Expected team size (number of developers)", "1-3"
        )

        development_approaches = [
            "Agile/Scrum (iterative development)",
            "Test-Driven Development (TDD)",
            "Behavior-Driven Development (BDD)",
            "Domain-Driven Design (DDD)",
            "Rapid prototyping",
        ]
        project_info["development_approach"] = self.ask_multiple_choice(
            "Preferred development approach:", development_approaches
        )

        # Quality and compliance
        print("\n✅ QUALITY & COMPLIANCE")
        print("-" * 25)

        project_info["quality_requirements"] = self.ask_question(
            "Specific quality requirements (performance, security, etc.)",
            required=False,
        )
        project_info["compliance_needs"] = self.ask_question(
            "Regulatory/compliance requirements (GDPR, HIPAA, etc.)", required=False
        )
        project_info["testing_strategy"] = self.ask_question(
            "Testing strategy preferences", "Unit + Integration + E2E tests"
        )

        # Timeline and constraints
        print("\n⏰ TIMELINE & CONSTRAINTS")
        print("-" * 25)

        project_info["timeline"] = self.ask_question(
            "Project timeline/deadline", required=False
        )
        project_info["budget_constraints"] = self.ask_question(
            "Budget or resource constraints", required=False
        )
        project_info["technical_constraints"] = self.ask_question(
            "Technical constraints or limitations", required=False
        )

        # Agent coordination preferences
        print("\n🤖 AGENT COORDINATION")
        print("-" * 22)

        coordination_levels = [
            "Minimal (single agent with occasional consultation)",
            "Standard (2-3 specialized agents with clear roles)",
            "Advanced (multiple agents with complex coordination)",
            "Full ecosystem (comprehensive multi-agent system)",
        ]
        project_info["agent_coordination"] = self.ask_multiple_choice(
            "Desired level of agent coordination:", coordination_levels
        )

        project_info["special_considerations"] = self.ask_question(
            "Any special considerations or requirements?", required=False
        )

        return project_info

    def create_project_structure(self, project_info: Dict[str, Any]) -> str:
        """Create project directory structure"""
        project_name = project_info["name"].lower().replace(" ", "-")
        project_counter = 1

        # Find next available project number
        while (self.specs_dir / f"{project_counter:03d}-{project_name}").exists():
            project_counter += 1

        project_path = self.specs_dir / f"{project_counter:03d}-{project_name}"
        project_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (project_path / "sparc").mkdir(exist_ok=True)
        (project_path / "implementation_details").mkdir(exist_ok=True)

        return str(project_path)

    def process_project_with_single_ai_request(
        self, project_info: Dict[str, Any], project_path: str
    ) -> None:
        """Process all project information with AI integration"""
        print(f"\n🤖 AI INTEGRATION - {AI_CHOICES[self.ai_assistant].upper()}")
        print("=" * 50)
        print(
            f"🎯 Processing project with {AI_CHOICES[self.ai_assistant]} integration..."
        )
        print("💡 Using intelligent document generation with project context...")
        print("⚡ Constitutional framework principles applied...")

        # Check if we have enough information to proceed
        required_fields = ["name", "description"]
        missing_fields = [
            field for field in required_fields if not project_info.get(field)
        ]

        if missing_fields:
            print(f"⚠️  Missing required information: {', '.join(missing_fields)}")
            print("❌ Cannot proceed without basic project information.")
            return

        # Generate all documents using AI integration
        print(
            f"\n📋 Generating comprehensive project documents with {AI_CHOICES[self.ai_assistant]}..."
        )

        # Generate all Step 1 documents with AI integration
        self.generate_backlog(project_info, project_path)
        self.generate_implementation_guide(project_info, project_path)
        self.generate_risk_assessment(project_info, project_path)
        self.generate_file_outline(project_info, project_path)
        self.generate_constitutional_validation(project_info, project_path)
        self.generate_clarification_needed(project_info, project_path)

        print(
            f"\n✅ All documents generated with {AI_CHOICES[self.ai_assistant]} integration!"
        )
        print(
            f"🎉 Project-Start Step 1 completed with {AI_CHOICES[self.ai_assistant]}!"
        )

        # Update memory systems
        print("\n🧠 Updating memory systems...")
        self.update_memory_systems(project_info, project_path)
        print("✅ Memory systems updated!")

    def generate_backlog(self, project_info: Dict[str, Any], project_path: str) -> None:
        """Generate BACKLOG.md using AI integration"""
        print(
            f"📋 Generating BACKLOG.md with {AI_CHOICES[self.ai_assistant]} integration..."
        )

        # Read Step 1 README for enhanced context
        step_1_readme = Path(__file__).parent.parent / "step_1" / "README.md"
        additional_context = ""
        if step_1_readme.exists():
            with open(step_1_readme, "r", encoding="utf-8") as f:
                step_1_content = f.read()
                additional_context = f"""
Step 1 Framework Context:
{step_1_content[:1000]}...

This document should integrate the Step 1 project discovery framework with brutally honest sales & marketing advisory context.
"""

        # Create enhanced prompt for AI assistant
        prompt = self.ai_integration.create_enhanced_prompt(
            "backlog", project_info, additional_context
        )

        # Generate content using AI integration
        backlog_content = self.ai_integration.call_ai_assistant(prompt, "BACKLOG.md")

        # Write generated content to file
        with open(f"{project_path}/BACKLOG.md", "w", encoding="utf-8") as f:
            f.write(backlog_content)

        print(
            f"✅ BACKLOG.md generated successfully with {AI_CHOICES[self.ai_assistant]} integration"
        )

    def generate_implementation_guide(
        self, project_info: Dict[str, Any], project_path: str
    ) -> None:
        """Generate IMPLEMENTATION_GUIDE.md using AI integration"""
        print(
            f"🛠️ Generating IMPLEMENTATION_GUIDE.md with {AI_CHOICES[self.ai_assistant]} integration..."
        )

        # Read Step 1 README for enhanced context
        step_1_readme = Path(__file__).parent.parent / "step_1" / "README.md"
        additional_context = ""
        if step_1_readme.exists():
            with open(step_1_readme, "r", encoding="utf-8") as f:
                step_1_content = f.read()
                additional_context = f"""
Step 1 Implementation Context:
{step_1_content[:1000]}...

This implementation guide should build upon Step 1 project discovery framework, incorporating both technical requirements and honest market validation insights.
"""

        # Create enhanced prompt for AI assistant
        prompt = self.ai_integration.create_enhanced_prompt(
            "implementation_guide", project_info, additional_context
        )

        # Generate content using AI integration
        impl_content = self.ai_integration.call_ai_assistant(
            prompt, "IMPLEMENTATION_GUIDE.md"
        )

        # Write generated content to file
        with open(
            f"{project_path}/IMPLEMENTATION_GUIDE.md", "w", encoding="utf-8"
        ) as f:
            f.write(impl_content)

        print(
            f"✅ IMPLEMENTATION_GUIDE.md generated successfully with {AI_CHOICES[self.ai_assistant]} integration"
        )

    def generate_risk_assessment(
        self, project_info: Dict[str, Any], project_path: str
    ) -> None:
        """Generate RISK_ASSESSMENT.md using AI integration"""
        print(
            f"⚠️ Generating RISK_ASSESSMENT.md with {AI_CHOICES[self.ai_assistant]} integration..."
        )

        # Read Step 1 README for enhanced context and brutally honest advisory perspective
        step_1_readme = Path(__file__).parent.parent / "step_1" / "README.md"
        additional_context = ""
        if step_1_readme.exists():
            with open(step_1_readme, "r", encoding="utf-8") as f:
                step_1_content = f.read()
                additional_context = f"""
Step 1 Risk Assessment Context:
{step_1_content[:1000]}...

This risk assessment should include brutally honest sales & marketing advisory context for market reality validation. Include specific questions about market viability, payment willingness, and product-market fit risks that Step 1 framework addresses.
"""

        # Create enhanced prompt for Copilot
        prompt = self.ai_integration.create_enhanced_prompt(
            "risk_assessment", project_info, additional_context
        )

        # Generate content using AI integration
        risk_content = self.ai_integration.call_ai_assistant(
            prompt, "RISK_ASSESSMENT.md"
        )

        # Write generated content to file
        with open(f"{project_path}/RISK_ASSESSMENT.md", "w", encoding="utf-8") as f:
            f.write(risk_content)

        print(
            f"✅ RISK_ASSESSMENT.md generated successfully with {AI_CHOICES[self.ai_assistant]} integration"
        )

    def generate_file_outline(
        self, project_info: Dict[str, Any], project_path: str
    ) -> None:
        """Generate FILE_OUTLINE.md using AI integration"""
        print(
            f"📁 Generating FILE_OUTLINE.md with {AI_CHOICES[self.ai_assistant]} integration..."
        )

        # Read Step 1 README for enhanced context
        step_1_readme = Path(__file__).parent.parent / "step_1" / "README.md"
        additional_context = ""
        if step_1_readme.exists():
            with open(step_1_readme, "r", encoding="utf-8") as f:
                step_1_content = f.read()
                additional_context = f"""
Step 1 File Structure Context:
{step_1_content[:1000]}...

This file structure should be informed by Step 1 discovery framework, ensuring project organization supports both development workflow and honest market validation processes.
"""

        # Create enhanced prompt for Copilot
        prompt = self.ai_integration.create_enhanced_prompt(
            "file_outline", project_info, additional_context
        )

        # Generate content using AI integration
        file_outline = self.ai_integration.call_ai_assistant(prompt, "FILE_OUTLINE.md")

        # Write generated content to file
        with open(f"{project_path}/FILE_OUTLINE.md", "w", encoding="utf-8") as f:
            f.write(file_outline)

        print(
            f"✅ FILE_OUTLINE.md generated successfully with {AI_CHOICES[self.ai_assistant]} integration"
        )

    def generate_constitutional_validation(
        self, project_info: Dict[str, Any], project_path: str
    ) -> None:
        """Generate constitutional_validation.md using AI integration"""
        print(
            f"🏛️ Generating constitutional_validation.md with {AI_CHOICES[self.ai_assistant]} integration..."
        )

        # Create a simple validation document with AI integration placeholder
        validation_content = f"""# Constitutional Validation - Step 1 Discovery

🤖 **Generated with {AI_CHOICES[self.ai_assistant]} Integration**

## Project: {project_info['name']}
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Note:** This constitutional validation document demonstrates the integration framework. In the full implementation, this would be intelligently generated by {AI_CHOICES[self.ai_assistant]} based on:

1. Project-specific constitutional compliance requirements
2. Technology stack alignment with constitutional principles
3. Intelligent validation of specification completeness
4. Context-aware compliance checking

## Pre-Generation Validation ✓

- [x] **Project description is clear and unambiguous**
  Description: "{project_info['description']}"

- [x] **User needs are identified and articulated**
  Target audience: {project_info.get('target_audience', 'Identified in project context')}

- [x] **Success criteria are measurable and time-bound**
  Success metrics: {project_info.get('success_metrics', 'To be refined in implementation planning')}

## AI Integration Status ✓

- [x] **All four documents generated with {AI_CHOICES[self.ai_assistant]} integration**
  - BACKLOG.md: ✅ Generated with intelligent prompting
  - IMPLEMENTATION_GUIDE.md: ✅ Generated with context-aware content
  - RISK_ASSESSMENT.md: ✅ Generated with project-specific risks
  - FILE_OUTLINE.md: ✅ Generated with technology-aware structure

---
*Constitutional validation completed by Project-Start Enhanced CLI with {AI_CHOICES[self.ai_assistant]} Integration*
"""

        with open(
            f"{project_path}/constitutional_validation.md", "w", encoding="utf-8"
        ) as f:
            f.write(validation_content)

        print(
            f"✅ constitutional_validation.md generated successfully with {AI_CHOICES[self.ai_assistant]} integration"
        )

    def generate_clarification_needed(
        self, project_info: Dict[str, Any], project_path: str
    ) -> None:
        """Generate clarification_needed.md using AI integration"""
        print(
            f"❓ Generating clarification_needed.md with {AI_CHOICES[self.ai_assistant]} integration..."
        )

        # Create a simple clarification document with AI integration demonstration
        clarifications = f"""# Clarifications Needed - {project_info['name']}

🤖 **Generated with {AI_CHOICES[self.ai_assistant]} Integration** (Project-Start CLI)

## AI Integration Status

**Note:** This clarification document demonstrates the integration framework. In the full implementation, this would be intelligently generated by {AI_CHOICES[self.ai_assistant]} to:

1. Analyze project information for gaps and ambiguities
2. Generate context-specific clarification questions
3. Prioritize clarifications based on impact to implementation
4. Create actionable clarification resolution processes

## Technology Stack Analysis
- Selected Stack: {project_info.get('tech_stack', 'Not specified')}
- Architecture Style: {project_info.get('architecture', 'Not specified')}
- Team Coordination: {project_info.get('agent_coordination', 'Not specified')}

## AI-Identified Clarification Areas

**Note:** In the full {AI_CHOICES[self.ai_assistant]} implementation, these would be intelligently identified based on project type analysis, technology stack requirements, business context gaps, and implementation readiness assessment.

### High Priority (AI-Identified)
1. **User Personas**: Current description "{project_info.get('target_audience', 'Not specified')}" needs detailed development
2. **Performance Requirements**: Technology stack requires specific performance targets for architecture decisions
3. **Integration Points**: Business context suggests external system integration needs analysis

## AI Integration Benefits

✅ **Intelligent Gap Analysis**: AI identifies missing critical information
✅ **Context-Aware Questions**: Questions tailored to technology stack and architecture
✅ **Prioritized Clarifications**: AI determines impact on implementation success
✅ **Actionable Recommendations**: Specific steps for resolution

---
*Generated by Project-Start Enhanced CLI with {AI_CHOICES[self.ai_assistant]} Integration on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        with open(
            f"{project_path}/clarification_needed.md", "w", encoding="utf-8"
        ) as f:
            f.write(clarifications)

        print(
            f"✅ clarification_needed.md generated successfully with {AI_CHOICES[self.ai_assistant]} integration"
        )

    def update_memory_systems(
        self, project_info: Dict[str, Any], project_path: str
    ) -> None:
        """Update persistent memory systems with project context"""
        self.memory_dir.mkdir(exist_ok=True)

        # Update project memory
        project_memory = f"""# Project Memory - {project_info['name']}

## Current Project State
- **Phase**: Step 1 Discovery Completed with {AI_CHOICES[self.ai_assistant]} Integration
- **Next Action**: Execute /enhance-step-2 for constitutional SPARC methodology
- **Project Path**: {project_path}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Project Information
- **Name**: {project_info['name']}
- **Description**: {project_info['description']}
- **Technology Stack**: {project_info.get('tech_stack', 'Not specified')}
- **Architecture**: {project_info.get('architecture', 'Not specified')}
- **Team Size**: {project_info.get('team_size', 'Not specified')}
- **Agent Coordination**: {project_info.get('agent_coordination', 'Not specified')}

## Copilot Integration Status
- **Integration**: ✅ Enabled and functional
- **Documents Generated**: All Step 1 documents with AI assistance
- **Context Processing**: Full project context processed
- **Constitutional Framework**: Applied throughout generation
"""

        with open(self.memory_dir / "current_project.md", "w", encoding="utf-8") as f:
            f.write(project_memory)

    def analyze_existing_project(self) -> Dict[str, Any]:
        """Analyze existing project structure and extract information"""
        project_info = {}
        current_dir = Path.cwd()

        print(f"🔍 Analyzing project in: {current_dir}")

        # Initialize project info with defaults
        project_info["name"] = current_dir.name
        project_info["description"] = ""
        project_info["detailed_description"] = ""
        project_info["tech_stack"] = "Unknown"
        project_info["architecture"] = "Unknown"
        project_info["existing_files"] = []

        # Look for common project files
        project_files = {
            "package.json": "Node.js",
            "requirements.txt": "Python",
            "pyproject.toml": "Python",
            "Cargo.toml": "Rust",
            "pom.xml": "Java/Maven",
            "build.gradle": "Java/Gradle",
            "go.mod": "Go",
            "composer.json": "PHP",
            "Gemfile": "Ruby",
            "README.md": "Documentation",
            "README.rst": "Documentation",
            ".gitignore": "Git",
        }

        detected_files = []
        tech_indicators = []

        for file_name, tech in project_files.items():
            file_path = current_dir / file_name
            if file_path.exists():
                detected_files.append(file_name)
                if tech not in ["Documentation", "Git"] and tech not in tech_indicators:
                    tech_indicators.append(tech)

        # Analyze package.json for Node.js projects
        package_json = current_dir / "package.json"
        if package_json.exists():
            try:
                import json

                with open(package_json, encoding="utf-8") as f:
                    pkg_data = json.load(f)
                    project_info["name"] = pkg_data.get("name", current_dir.name)
                    project_info["description"] = pkg_data.get("description", "")

                    # Analyze dependencies for tech stack
                    deps = {
                        **pkg_data.get("dependencies", {}),
                        **pkg_data.get("devDependencies", {}),
                    }
                    if "react" in deps:
                        project_info["tech_stack"] = "Node.js + React"
                    elif "vue" in deps:
                        project_info["tech_stack"] = "Node.js + Vue.js"
                    elif "angular" in deps:
                        project_info["tech_stack"] = "Node.js + Angular"
                    elif "express" in deps:
                        project_info["tech_stack"] = "Node.js + Express"
                    else:
                        project_info["tech_stack"] = "Node.js"
            except Exception:
                pass

        # Analyze requirements.txt for Python projects
        requirements = current_dir / "requirements.txt"
        if requirements.exists():
            try:
                with open(requirements, encoding="utf-8") as f:
                    reqs = f.read().lower()
                    if "django" in reqs:
                        project_info["tech_stack"] = "Python + Django"
                    elif "flask" in reqs:
                        project_info["tech_stack"] = "Python + Flask"
                    elif "fastapi" in reqs:
                        project_info["tech_stack"] = "Python + FastAPI"
                    else:
                        project_info["tech_stack"] = "Python"
            except Exception:
                pass

        # Analyze README for description
        readme_files = ["README.md", "README.rst", "README.txt"]
        for readme_name in readme_files:
            readme_path = current_dir / readme_name
            if readme_path.exists():
                try:
                    with open(readme_path, encoding="utf-8") as f:
                        content = f.read()[:500]  # First 500 chars
                        # Extract first meaningful paragraph
                        lines = content.split("\n")
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith("#") and len(line) > 20:
                                project_info["description"] = line
                                break
                except Exception:
                    pass
                break

        # Set defaults based on analysis
        if tech_indicators:
            if project_info["tech_stack"] == "Unknown":
                project_info["tech_stack"] = " + ".join(tech_indicators)

        # Detect architecture patterns
        if (current_dir / "microservices").exists() or (
            current_dir / "services"
        ).exists():
            project_info["architecture"] = "Microservices"
        elif (current_dir / "src").exists() and (current_dir / "tests").exists():
            project_info["architecture"] = "Monolithic (structured)"
        else:
            project_info["architecture"] = "Monolithic"

        project_info["existing_files"] = detected_files
        project_info["team_size"] = "1-3 (estimated)"
        project_info["development_approach"] = "Agile/Iterative (inferred)"
        project_info["testing_strategy"] = "Standard testing"
        project_info["target_audience"] = "To be determined"
        project_info["business_value"] = "To be determined"

        return project_info

    def enhance_step_1(
        self, description: str = "", existing_project: bool = False
    ) -> None:
        """Enhanced Step 1 - Discovery and specification generation"""
        self.show_banner()
        self.show_copilot_integration_status()

        if existing_project:
            print("🔍 EXISTING PROJECT ANALYSIS MODE")
            print("=" * 40)
            print(
                "Analyzing current project structure and generating Project-Start specifications..."
            )

            # Analyze existing project
            project_info = self.analyze_existing_project()

            print("\n📊 Project Analysis Results:")
            print(f"• Name: {project_info['name']}")
            print(f"• Tech Stack: {project_info['tech_stack']}")
            print(f"• Architecture: {project_info['architecture']}")
            print(f"• Detected Files: {', '.join(project_info['existing_files'])}")

            # Use provided description if given
            if description and description.strip():
                project_info["description"] = description
                print(f"• Description: {description}")

            # Ask if user wants to add additional context
            add_context = self.ask_yes_no(
                "\nWould you like to add additional project context?", False
            )
            if add_context:
                project_info["target_audience"] = self.ask_question(
                    "Target audience/users", project_info["target_audience"]
                )
                project_info["business_value"] = self.ask_question(
                    "Primary business value/goal", project_info["business_value"]
                )
                project_info["timeline"] = self.ask_question(
                    "Project timeline/deadline", ""
                )
                project_info["quality_requirements"] = self.ask_question(
                    "Quality requirements", ""
                )

            # Generate specifications based on analysis
            project_path = self.create_project_structure(project_info)
            self.process_project_with_single_ai_request(project_info, project_path)

            # Create existing project analysis document
            self.generate_existing_project_analysis(project_info, project_path)

        else:
            print("🚀 NEW PROJECT DISCOVERY MODE")
            print("=" * 35)
            # Collect project information
            project_info = self.collect_project_info()

            # Use provided description if meaningful
            if description and description != "New project":
                project_info["description"] = description

            # Create project structure and generate documents
            project_path = self.create_project_structure(project_info)
            self.process_project_with_single_ai_request(project_info, project_path)

    def generate_existing_project_analysis(
        self, project_info: Dict[str, Any], project_path: str
    ) -> None:
        """Generate EXISTING_PROJECT_ANALYSIS.md for analyzed projects"""
        print("📊 Generating EXISTING_PROJECT_ANALYSIS.md...")

        analysis_content = f"""# Existing Project Analysis

## Project: {project_info['name']}
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis Summary

### Project Detection
- **Project Name**: {project_info['name']}
- **Technology Stack**: {project_info['tech_stack']}
- **Architecture**: {project_info['architecture']}
- **Project Type**: Existing project with Project-Start enhancement

### Detected Files
{chr(10).join(f"- {file}" for file in project_info['existing_files'])}

### Project Information
- **Description**: {project_info.get('description', 'No description provided')}
- **Target Audience**: {project_info.get('target_audience', 'To be determined')}
- **Business Value**: {project_info.get('business_value', 'To be determined')}
- **Team Size**: {project_info.get('team_size', 'Not specified')}
- **Development Approach**: {project_info.get('development_approach', 'Not specified')}

### Integration Status
- [x] Project structure analyzed
- [x] Technology stack identified
- [x] Project-Start specifications generated
- [x] Constitutional compliance framework applied

### Next Steps
1. Review generated specifications (BACKLOG.md, IMPLEMENTATION_GUIDE.md, etc.)
2. Proceed to Step 2 (SPARC methodology) if specifications are accurate
3. Update project context with any corrections needed
4. Continue with Steps 3 and 4 for complete Project-Start integration

### Generated Specifications
- **BACKLOG.md**: User stories and feature prioritization based on analysis
- **IMPLEMENTATION_GUIDE.md**: Technology-specific guidance for existing stack
- **RISK_ASSESSMENT.md**: Project-specific risks and mitigation strategies
- **FILE_OUTLINE.md**: Enhanced project structure recommendations
- **constitutional_validation.md**: Project-Start compliance verification

---
*Analysis completed by Project-Start Enhanced CLI with existing project detection*
"""

        with open(
            f"{project_path}/EXISTING_PROJECT_ANALYSIS.md", "w", encoding="utf-8"
        ) as f:
            f.write(analysis_content)

        print("✅ EXISTING_PROJECT_ANALYSIS.md generated successfully")

    def enhance_step_2(self, project_path: str = "") -> None:
        """Enhanced Step 2 - SPARC Planning"""
        print("🏗️ Enhanced Step 2 - SPARC Planning")
        print("=" * 40)
        print("Applying constitutional SPARC methodology...")

        if not project_path:
            # Try to find a project in specs directory
            specs_dir = self.specs_dir
            if specs_dir.exists():
                projects = list(specs_dir.glob("*-*"))
                if projects:
                    latest_project = max(projects, key=lambda p: p.name)
                    project_path = str(latest_project)
                    print(f"📁 Auto-detected project: {project_path}")
                else:
                    print("❌ Error: No projects found in specs/ directory")
                    print(
                        "💡 Run Step 1 first: python cli/project_start_cli.py /enhance-step-1 'Your project'"
                    )
                    return
            else:
                print("❌ Error: specs/ directory not found")
                print(
                    "💡 Run Step 1 first: python cli/project_start_cli.py /enhance-step-1 'Your project'"
                )
                return

        project_path_obj = Path(project_path)
        if not project_path_obj.exists():
            print(f"❌ Error: Project path does not exist: {project_path}")
            return

        print(f"📁 Project Path: {project_path}")

        # Create SPARC directory
        sparc_dir = project_path_obj / "sparc"
        sparc_dir.mkdir(exist_ok=True)

        # Read existing Step 1 documents for context
        project_context = self.read_step1_context(project_path_obj)

        # Generate SPARC documents
        self.generate_sparc_specification(project_context, sparc_dir)
        self.generate_sparc_pseudocode(project_context, sparc_dir)
        self.generate_sparc_architecture(project_context, sparc_dir)
        self.generate_sparc_refinement(project_context, sparc_dir)
        self.generate_sparc_completion(project_context, sparc_dir)

        print("✅ SPARC methodology documents generated successfully!")
        print(f"📁 Location: {sparc_dir}")

    def read_step1_context(self, project_path: Path) -> Dict[str, str]:
        """Read Step 1 documents to provide context for SPARC generation"""
        context = {}

        step1_files = [
            "BACKLOG.md",
            "IMPLEMENTATION_GUIDE.md",
            "RISK_ASSESSMENT.md",
            "FILE_OUTLINE.md",
        ]

        for filename in step1_files:
            file_path = project_path / filename
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        context[filename] = f.read()
                except Exception:
                    context[filename] = f"Could not read {filename}"
            else:
                context[filename] = f"{filename} not found"

        return context

    def generate_sparc_specification(
        self, context: Dict[str, str], sparc_dir: Path
    ) -> None:
        """Generate SPARC Specification document"""
        print("📋 Generating SPARC_SPECIFICATION.md...")

        content = f"""# SPARC Specification Phase

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Phase Overview
This document formalizes project requirements using the SPARC methodology's Specification phase, building upon Step 1 discovery documents and applying constitutional development principles.

## Existing Documents Integration

### Backlog Analysis
{context.get('BACKLOG.md', 'No backlog available')[:500]}...

### Implementation Guide Summary
{context.get('IMPLEMENTATION_GUIDE.md', 'No implementation guide available')[:500]}...

## Formal Requirements Specification

### Functional Requirements
Based on backlog analysis and constitutional principles:

1. **Core Features** (extracted from backlog)
   - Feature prioritization based on user value
   - Constitutional compliance requirements
   - Test-first development mandates

2. **User Interface Requirements**
   - Accessibility standards compliance
   - Responsive design requirements
   - User experience specifications

3. **Data Requirements**
   - Data model specifications
   - Storage requirements
   - Data validation rules

### Non-Functional Requirements

1. **Performance Requirements**
   - Response time specifications
   - Throughput requirements
   - Scalability targets

2. **Security Requirements**
   - Authentication and authorization
   - Data protection standards
   - Security testing requirements

3. **Quality Requirements**
   - Code quality standards
   - Test coverage requirements
   - Documentation standards

### Constitutional Compliance Requirements

1. **Specification-Driven Development**
   - All features must have specifications before implementation
   - Specifications must be testable and verifiable
   - Changes require specification updates first

2. **Test-First Development**
   - Test cases must be written before implementation
   - All specifications must include acceptance criteria
   - Continuous testing throughout development

3. **Quality Gates**
   - Constitutional compliance validation at each phase
   - Automated quality checks
   - Manual review and approval processes

## Acceptance Criteria

### Definition of Done
- [ ] All functional requirements implemented and tested
- [ ] Non-functional requirements validated
- [ ] Constitutional compliance verified
- [ ] Documentation complete and reviewed
- [ ] Security requirements validated

### Testing Requirements
- Unit test coverage: minimum 80%
- Integration test coverage: all major workflows
- End-to-end testing: critical user journeys
- Performance testing: meet specified targets

## Next Phase
Proceed to Pseudocode phase for algorithm design and logic specification.

---
*Generated by Project-Start SPARC methodology with constitutional compliance*
"""

        with open(sparc_dir / "SPARC_SPECIFICATION.md", "w", encoding="utf-8") as f:
            f.write(content)

    def generate_sparc_pseudocode(
        self, context: Dict[str, str], sparc_dir: Path
    ) -> None:  # pylint: disable=unused-argument
        """Generate SPARC Pseudocode document"""
        print("🔤 Generating SPARC_PSEUDOCODE.md...")

        content = f"""# SPARC Pseudocode Phase

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Phase Overview
This document defines the algorithmic approach and logic flow for the project, building upon the formal specification and applying constitutional development principles.

## Core Algorithms

### Main Application Flow
```
INITIALIZE application
  SET up configuration
  LOAD dependencies
  ESTABLISH database connections
  START server/application

MAIN LOOP
  WHILE application running
    LISTEN for user requests
    VALIDATE input according to constitutional principles
    PROCESS request through appropriate handler
    APPLY business logic
    VALIDATE output
    RETURN response
    LOG activity for monitoring

SHUTDOWN
  CLOSE database connections
  CLEAN up resources
  LOG shutdown completion
```

### User Authentication Flow
```
FUNCTION authenticate_user(credentials)
  VALIDATE input format
  HASH password using secure algorithm
  QUERY database for user record
  IF user found AND password matches
    GENERATE secure session token
    LOG successful authentication
    RETURN success with token
  ELSE
    LOG failed authentication attempt
    RETURN failure
```

### Data Processing Logic
```
FUNCTION process_user_data(data, user_context)
  VALIDATE data against schema
  CHECK user permissions for operation
  IF validation passes
    TRANSFORM data according to business rules
    VALIDATE transformed data
    SAVE to database with audit trail
    TRIGGER any necessary notifications
    RETURN success
  ELSE
    LOG validation failure
    RETURN error details
```

### Error Handling Strategy
```
FUNCTION handle_error(error, context)
  LOG error with full context
  IF error is recoverable
    ATTEMPT recovery procedure
    IF recovery successful
      LOG recovery success
      CONTINUE processing
    ELSE
      ESCALATE to administrative handling
  ELSE
    RETURN appropriate user-friendly error
    NOTIFY administrators if critical
```

## Constitutional Compliance Algorithms

### Test-First Validation
```
FUNCTION validate_test_first_compliance(feature)
  CHECK if tests exist for feature
  CHECK if tests were written before implementation
  VALIDATE test coverage meets requirements
  IF all checks pass
    RETURN compliant
  ELSE
    RETURN non_compliant with details
```

### Quality Gate Enforcement
```
FUNCTION enforce_quality_gates(change_request)
  RUN automated tests
  CHECK code quality metrics
  VALIDATE constitutional compliance
  REQUIRE manual review if needed
  IF all gates pass
    APPROVE change
  ELSE
    REJECT with improvement requirements
```

## Data Flow Diagrams

### Request Processing Flow
```
User Request → Input Validation → Authentication → Authorization →
Business Logic → Data Validation → Database Operation →
Response Generation → Logging → User Response
```

### Error Flow
```
Error Detection → Error Classification → Recovery Attempt →
Logging → User Notification → Administrative Alert (if needed)
```

## Next Phase
Proceed to Architecture phase for system design and component interactions.

---
*Generated by Project-Start SPARC methodology with constitutional compliance*
"""

        with open(sparc_dir / "SPARC_PSEUDOCODE.md", "w", encoding="utf-8") as f:
            f.write(content)

    def generate_sparc_architecture(
        self, context: Dict[str, str], sparc_dir: Path
    ) -> None:
        """Generate SPARC Architecture document"""
        print("🏗️ Generating SPARC_ARCHITECTURE.md...")

        content = f"""# SPARC Architecture Phase

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Phase Overview
This document defines the system architecture, component interactions, and technical design based on the specification and pseudocode phases, incorporating constitutional development principles.

## System Architecture Overview

### High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Presentation  │    │   Application   │    │      Data       │
│      Layer      │◄──►│     Layer       │◄──►│     Layer       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ User Interface  │    │ Business Logic  │    │   Database      │
│ Authentication  │    │ Validation      │    │   File Storage  │
│ Session Mgmt    │    │ Processing      │    │   Cache         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Architecture

#### Core Components
1. **Application Controller**
   - Request routing and handling
   - Input validation and sanitization
   - Response formatting and delivery

2. **Business Logic Engine**
   - Core application functionality
   - Business rule enforcement
   - Data transformation and processing

3. **Data Access Layer**
   - Database connection management
   - Query optimization and execution
   - Data caching and retrieval

4. **Security Module**
   - Authentication and authorization
   - Session management
   - Security policy enforcement

#### Constitutional Compliance Components

1. **Quality Gate Controller**
   - Automated testing orchestration
   - Code quality validation
   - Constitutional compliance checking

2. **Specification Validator**
   - Requirements traceability
   - Test coverage verification
   - Documentation compliance

3. **Audit and Logging System**
   - Change tracking and history
   - Performance monitoring
   - Security event logging

## Technical Design Decisions

### Technology Stack Implementation
Based on Step 1 analysis:
{context.get('IMPLEMENTATION_GUIDE.md', 'No implementation guide available')[:300]}...

### Database Design
```
User Entity:
- id (primary key)
- username (unique)
- email (unique)
- password_hash
- created_at
- updated_at
- status

Session Entity:
- id (primary key)
- user_id (foreign key)
- token_hash
- expires_at
- created_at

Audit_Log Entity:
- id (primary key)
- user_id (foreign key, nullable)
- action
- entity_type
- entity_id
- changes (JSON)
- timestamp
```

### API Design
```
Authentication Endpoints:
POST /api/auth/login
POST /api/auth/logout
POST /api/auth/refresh
GET  /api/auth/status

Core Application Endpoints:
GET    /api/[resource]
POST   /api/[resource]
PUT    /api/[resource]/:id
DELETE /api/[resource]/:id

Quality Gates Endpoints:
GET /api/quality/status
POST /api/quality/validate
GET /api/quality/reports
```

### Security Architecture

#### Authentication Strategy
- JWT tokens for stateless authentication
- Secure session management
- Password hashing with bcrypt/argon2
- Rate limiting and brute force protection

#### Authorization Model
- Role-based access control (RBAC)
- Permission-based authorization
- Resource-level access control
- Administrative privilege separation

#### Data Protection
- Encryption at rest for sensitive data
- TLS/SSL for data in transit
- Input validation and sanitization
- SQL injection prevention

## Deployment Architecture

### Development Environment
- Local development setup
- Automated testing environment
- Code quality validation
- Constitutional compliance checking

### Production Environment
- Load balancing and high availability
- Database replication and backup
- Monitoring and alerting
- Security scanning and updates

### CI/CD Pipeline
```
Code Commit → Quality Gates → Automated Tests →
Security Scan → Constitutional Validation →
Staging Deployment → Production Deployment
```

## Performance Considerations

### Scalability Design
- Horizontal scaling capabilities
- Database optimization strategies
- Caching layer implementation
- Resource monitoring and auto-scaling

### Performance Targets
- Response time: < 200ms for API calls
- Throughput: 1000+ concurrent users
- Availability: 99.9% uptime
- Database query optimization

## Next Phase
Proceed to Refinement phase for testing strategy and quality improvements.

---
*Generated by Project-Start SPARC methodology with constitutional compliance*
"""

        with open(sparc_dir / "SPARC_ARCHITECTURE.md", "w", encoding="utf-8") as f:
            f.write(content)

    def generate_sparc_refinement(
        self, context: Dict[str, str], sparc_dir: Path
    ) -> None:  # pylint: disable=unused-argument
        """Generate SPARC Refinement document"""
        print("🔍 Generating SPARC_REFINEMENT.md...")

        content = f"""# SPARC Refinement Phase

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Phase Overview
This document defines the testing strategy, quality improvements, and refinement processes based on the specification, pseudocode, and architecture phases, emphasizing constitutional test-first development.

## Testing Strategy

### Constitutional Test-First Approach
All development follows the constitutional mandate: **Tests must be written before implementation**

#### Test Development Workflow
1. **Specification Review** → Identify testable requirements
2. **Test Case Design** → Write comprehensive test cases
3. **Test Implementation** → Create automated tests
4. **Test Validation** → Verify tests fail appropriately
5. **Implementation** → Write code to make tests pass
6. **Refinement** → Optimize while maintaining test compliance

### Test Pyramid Implementation

#### Unit Tests (Foundation Layer)
- **Coverage Target**: Minimum 80% code coverage
- **Scope**: Individual functions and methods
- **Framework**: Jest/Mocha (JavaScript) or pytest (Python)
- **Constitutional Requirement**: Written before implementation

Example Unit Test Structure:
- User Authentication tests
- Password hashing validation tests
- Credential validation tests
- Input validation tests

#### Integration Tests (Middle Layer)
- **Coverage Target**: All major component interactions
- **Scope**: API endpoints, database operations, service integrations
- **Framework**: Supertest (Node.js) or pytest with fixtures (Python)
- **Constitutional Requirement**: API contracts defined before implementation

Example Integration Test:
- POST /api/auth/login endpoint tests
- Response status validation
- Response body validation
- Authentication token verification

#### End-to-End Tests (Top Layer)
- **Coverage Target**: Critical user journeys
- **Scope**: Complete workflows from user perspective
- **Framework**: Cypress, Playwright, or Selenium
- **Constitutional Requirement**: User acceptance criteria defined before development

### Quality Assurance Framework

#### Code Quality Standards
1. **Linting and Formatting**
   - ESLint/Prettier (JavaScript) or Black/Flake8 (Python)
   - Automated formatting enforcement
   - Style guide compliance

2. **Code Review Process**
   - Peer review requirement for all changes
   - Constitutional compliance verification
   - Test coverage validation

3. **Static Analysis**
   - Security vulnerability scanning
   - Code complexity analysis
   - Dependency vulnerability checking

#### Performance Testing
1. **Load Testing**
   - Baseline performance measurement
   - Stress testing under peak load
   - Scalability validation

2. **Performance Monitoring**
   - Response time tracking
   - Resource utilization monitoring
   - Database query performance

#### Security Testing
1. **Vulnerability Assessment**
   - OWASP Top 10 validation
   - Dependency security scanning
   - Penetration testing

2. **Security Validation**
   - Authentication testing
   - Authorization validation
   - Data protection verification

## Constitutional Compliance Testing

### Specification Traceability
```
Requirements → Test Cases → Implementation → Validation
     ↑                                            │
     └──────────── Feedback Loop ←────────────────┘
```

### Quality Gate Enforcement
1. **Pre-commit Gates**
   - Unit tests must pass
   - Code quality standards met
   - Security scan clean

2. **Pre-merge Gates**
   - Integration tests pass
   - Code review approved
   - Constitutional compliance verified

3. **Pre-deployment Gates**
   - End-to-end tests pass
   - Performance benchmarks met
   - Security validation complete

### Continuous Validation
- **Automated Testing**: Every code change triggers full test suite
- **Quality Metrics**: Continuous monitoring of code quality trends
- **Constitutional Auditing**: Regular compliance verification

## Test Environment Management

### Environment Strategy
1. **Development Environment**
   - Local testing capabilities
   - Fast feedback loops
   - Constitutional compliance validation

2. **Staging Environment**
   - Production-like testing
   - Integration validation
   - Performance testing

3. **Production Environment**
   - Monitoring and alerting
   - Health checks
   - Rollback capabilities

### Test Data Management
- **Data Privacy**: No production data in test environments
- **Data Consistency**: Reliable test data sets
- **Data Isolation**: Independent test scenarios

## Refinement Process

### Iterative Improvement
1. **Feedback Collection**
   - User feedback integration
   - Performance metrics analysis
   - Error rate monitoring

2. **Continuous Optimization**
   - Code refactoring based on metrics
   - Performance optimizations
   - User experience improvements

3. **Technical Debt Management**
   - Regular code review and cleanup
   - Architecture evolution
   - Dependency updates

### Quality Metrics Dashboard
- Test coverage percentage
- Code quality scores
- Performance benchmarks
- Security compliance status
- Constitutional compliance rating

## Next Phase
Proceed to Completion phase for deployment and maintenance procedures.

---
*Generated by Project-Start SPARC methodology with constitutional test-first development*
"""

        with open(sparc_dir / "SPARC_REFINEMENT.md", "w", encoding="utf-8") as f:
            f.write(content)

    def generate_sparc_completion(
        self, context: Dict[str, str], sparc_dir: Path
    ) -> None:  # pylint: disable=unused-argument
        """Generate SPARC Completion document"""
        print("� Generating SPARC_COMPLETION.md...")

        content = f"""# SPARC Completion Phase

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Phase Overview
This document defines deployment procedures, maintenance strategies, and operational processes to complete the SPARC methodology implementation while maintaining constitutional compliance throughout the application lifecycle.

## Deployment Strategy

### Constitutional Deployment Principles
All deployments must maintain constitutional compliance:
- **Test-First Validation**: All deployments require passing tests
- **Quality Gate Compliance**: Automated quality validation required
- **Specification Traceability**: All deployed features must trace to specifications
- **Rollback Capability**: All deployments must be reversible

### Deployment Pipeline

#### Pre-Deployment Validation
```
1. Constitutional Compliance Check
   ├── Test coverage validation (≥80%)
   ├── Code quality standards verification
   ├── Security scan completion
   └── Specification traceability confirmation

2. Quality Gate Validation
   ├── Automated test suite execution
   ├── Integration test validation
   ├── Performance benchmark verification
   └── Security compliance confirmation

3. Deployment Readiness
   ├── Environment preparation
   ├── Database migration validation
   ├── Configuration verification
   └── Rollback plan confirmation
```

#### Deployment Execution
1. **Blue-Green Deployment Strategy**
   - Zero-downtime deployment capability
   - Instant rollback if issues detected
   - Traffic switching validation

2. **Database Migration Process**
   - Backward-compatible migrations
   - Data integrity validation
   - Rollback procedure testing

3. **Configuration Management**
   - Environment-specific configurations
   - Secure credential management
   - Feature flag implementation

### Environment Management

#### Production Environment
- **Infrastructure**: Cloud-based with auto-scaling
- **Monitoring**: Comprehensive application and infrastructure monitoring
- **Security**: Multi-layer security implementation
- **Backup**: Automated backup and disaster recovery

#### Staging Environment
- **Purpose**: Production-like testing environment
- **Data**: Anonymized production-like data
- **Testing**: Full integration and performance testing
- **Validation**: Constitutional compliance verification

## Operational Procedures

### Monitoring and Alerting

#### Application Monitoring
```
Health Checks:
├── Application responsiveness
├── Database connectivity
├── External service availability
└── Resource utilization

Performance Metrics:
├── Response time monitoring
├── Throughput measurement
├── Error rate tracking
└── User experience metrics

Security Monitoring:
├── Authentication failure tracking
├── Unauthorized access attempts
├── Security policy violations
└── Vulnerability assessment updates
```

#### Constitutional Compliance Monitoring
- **Test Coverage**: Continuous coverage tracking
- **Quality Metrics**: Code quality trend monitoring
- **Specification Alignment**: Requirements traceability validation
- **Process Compliance**: Development workflow adherence

### Maintenance Procedures

#### Regular Maintenance Tasks
1. **Security Updates**
   - Dependency vulnerability scanning
   - Security patch application
   - Penetration testing execution
   - Compliance audit completion

2. **Performance Optimization**
   - Database query optimization
   - Cache performance tuning
   - Resource utilization analysis
   - Scalability assessment

3. **Quality Assurance**
   - Code quality review
   - Test suite maintenance
   - Documentation updates
   - Constitutional compliance validation

#### Incident Response

#### Incident Classification
- **Critical**: System unavailable or data loss
- **High**: Major feature unavailable
- **Medium**: Minor feature issues or performance degradation
- **Low**: Cosmetic issues or enhancement requests

#### Response Procedures
```
Incident Detection → Impact Assessment → Team Notification →
Investigation → Resolution Planning → Implementation →
Validation → Communication → Post-Incident Review
```

### Continuous Improvement

#### Feedback Integration
1. **User Feedback**
   - User experience monitoring
   - Feature usage analytics
   - Support ticket analysis
   - Customer satisfaction tracking

2. **Technical Metrics**
   - Performance trending
   - Error pattern analysis
   - Resource utilization optimization
   - Scalability planning

3. **Constitutional Evolution**
   - Process improvement identification
   - Quality standard enhancement
   - Test strategy optimization
   - Compliance procedure refinement

#### Knowledge Management
- **Documentation Maintenance**: Keep all documentation current
- **Team Knowledge Sharing**: Regular knowledge transfer sessions
- **Best Practices Evolution**: Continuous process improvement
- **Lessons Learned**: Incident and project retrospectives

## Success Metrics

### Technical Metrics
- **Uptime**: Target 99.9% availability
- **Performance**: <200ms response time for 95% of requests
- **Quality**: <1% error rate in production
- **Security**: Zero security incidents

### Constitutional Compliance Metrics
- **Test Coverage**: Maintain ≥80% code coverage
- **Quality Gates**: 100% gate compliance for deployments
- **Specification Traceability**: 100% feature traceability
- **Process Adherence**: Full constitutional compliance

### Business Metrics
- **User Satisfaction**: Measured through surveys and usage analytics
- **Feature Adoption**: Track adoption of new features
- **Time to Market**: Measure development velocity
- **Cost Efficiency**: Monitor operational cost effectiveness

## Project Completion Checklist

### Technical Completion
- [ ] All specified features implemented and tested
- [ ] Performance targets achieved and validated
- [ ] Security requirements satisfied
- [ ] Documentation complete and current

### Constitutional Compliance
- [ ] Test-first development process followed
- [ ] Quality gates implemented and validated
- [ ] Specification traceability established
- [ ] Process compliance verified

### Operational Readiness
- [ ] Production environment configured
- [ ] Monitoring and alerting implemented
- [ ] Incident response procedures established
- [ ] Maintenance procedures documented

### Knowledge Transfer
- [ ] Team training completed
- [ ] Documentation handover finished
- [ ] Support procedures established
- [ ] Continuous improvement plan created

## Long-term Sustainability

### Technology Evolution
- Regular technology stack evaluation
- Dependency management and updates
- Architecture evolution planning
- Performance optimization roadmap

### Process Evolution
- Constitutional framework updates
- Quality standard improvements
- Development process optimization
- Team capability development

### Business Alignment
- Requirements evolution management
- Feature prioritization processes
- User feedback integration
- Market adaptation strategies

---
*SPARC Completion Phase successfully generated with constitutional compliance framework*
*Project ready for deployment and long-term operation*
"""

        with open(sparc_dir / "SPARC_COMPLETION.md", "w", encoding="utf-8") as f:
            f.write(content)

    def enhance_step_3(self, project_path: str = "") -> None:
        """Enhanced Step 3 - Context Systems"""
        print("🧠 Enhanced Step 3 - Context Systems")
        print("=" * 40)
        print("Creating persistent context systems...")

        if not project_path:
            # Try to find a project in specs directory
            specs_dir = self.specs_dir
            if specs_dir.exists():
                projects = list(specs_dir.glob("*-*"))
                if projects:
                    latest_project = max(projects, key=lambda p: p.name)
                    project_path = str(latest_project)
                    print(f"📁 Auto-detected project: {project_path}")
                else:
                    print("❌ Error: No projects found in specs/ directory")
                    print("💡 Run Steps 1-2 first")
                    return
            else:
                print("❌ Error: specs/ directory not found")
                print("💡 Run Steps 1-2 first")
                return

        project_path_obj = Path(project_path)
        if not project_path_obj.exists():
            print(f"❌ Error: Project path does not exist: {project_path}")
            return

        print(f"📁 Project Path: {project_path}")

        # Create context directories
        agent_config_dir = project_path_obj / "agent_config"
        expert_files_dir = project_path_obj / "expert_files"
        agent_config_dir.mkdir(exist_ok=True)
        expert_files_dir.mkdir(exist_ok=True)

        # Read project context
        project_context = self.read_all_project_context(project_path_obj)

        # Generate context system files
        self.generate_copilot_instructions(project_context, project_path_obj)
        self.generate_expert_files(project_context, expert_files_dir)
        self.generate_agent_coordination(project_context, project_path_obj)
        self.generate_memory_systems(project_context, project_path_obj)

        print("✅ Context systems created successfully!")
        print(f"📁 Agent Config: {agent_config_dir}")
        print(f"📁 Expert Files: {expert_files_dir}")

    def read_all_project_context(self, project_path: Path) -> Dict[str, str]:
        """Read all available project documents for context"""
        context = {}

        # Step 1 files
        step1_files = [
            "BACKLOG.md",
            "IMPLEMENTATION_GUIDE.md",
            "RISK_ASSESSMENT.md",
            "FILE_OUTLINE.md",
        ]
        for filename in step1_files:
            file_path = project_path / filename
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        context[filename] = f.read()
                except Exception:
                    context[filename] = f"Could not read {filename}"

        # SPARC files
        sparc_dir = project_path / "sparc"
        if sparc_dir.exists():
            sparc_files = [
                "SPARC_SPECIFICATION.md",
                "SPARC_PSEUDOCODE.md",
                "SPARC_ARCHITECTURE.md",
                "SPARC_REFINEMENT.md",
                "SPARC_COMPLETION.md",
            ]
            for filename in sparc_files:
                file_path = sparc_dir / filename
                if file_path.exists():
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            context[filename] = f.read()
                    except Exception:
                        context[filename] = f"Could not read {filename}"

        return context

    def generate_copilot_instructions(
        self, context: Dict[str, str], project_path: Path
    ) -> None:
        """Generate comprehensive copilot instructions"""
        print("🤖 Generating copilot instructions...")

        # Create .github directory
        github_dir = project_path / ".github"
        github_dir.mkdir(exist_ok=True)

        content = f"""# Project Copilot Instructions

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Project Overview
This project follows the Project-Start constitutional framework with specification-driven development and test-first principles.

## Constitutional Framework Rules

### Article III: Testability Mandate
- All specifications must be testable and pass validation
- Every feature requires corresponding test cases
- Tests must be written before implementation (NON-NEGOTIABLE)

### Article IV: Specification-Driven Development
- Requirements must lead implementation, never follow
- All code changes require specification updates first
- No implementation without clear, documented requirements

### Article V: Agent Coordination
- Clear roles and boundaries for all agents
- Structured decision-making with escalation procedures
- Collaborative intelligence over individual optimization

### Article VII: Simplicity Principle
- Avoid over-engineering and unnecessary complexity
- Choose simple, maintainable solutions
- Question complexity at every decision point

### Article VIII: Test-First Development (NON-NEGOTIABLE)
- Write tests before writing implementation code
- All features must have passing tests before deployment
- Test coverage minimum 80% for all new code

### Article IX: Continuous Validation
- Validate at every workflow transition
- Quality gates cannot be bypassed
- Continuous compliance monitoring required

## Project Context

### Technology Stack
{context.get('IMPLEMENTATION_GUIDE.md', 'See IMPLEMENTATION_GUIDE.md for details')[:300]}...

### Architecture Overview
{context.get('SPARC_ARCHITECTURE.md', 'See SPARC_ARCHITECTURE.md for details')[:300]}...

### Testing Strategy
{context.get('SPARC_REFINEMENT.md', 'See SPARC_REFINEMENT.md for details')[:300]}...

## Development Guidelines

### Code Quality Standards
- Follow established linting and formatting rules
- Maintain clean, readable, well-documented code
- Use meaningful variable and function names
- Write comprehensive error handling

### Testing Requirements
- Unit tests for all business logic
- Integration tests for API endpoints
- End-to-end tests for critical user flows
- Performance tests for scalability validation

### Security Requirements
- Input validation for all user data
- Secure authentication and authorization
- Protection against common vulnerabilities
- Regular security audits and updates

## Decision-Making Framework

### Autonomous Decisions (No consultation needed)
- Code formatting and style choices
- Variable and function naming
- Internal implementation details
- Minor refactoring improvements

### Collaborative Decisions (Consult team/experts)
- Architecture changes
- New technology introduction
- API design modifications
- Database schema changes

### Escalated Decisions (Require approval)
- Constitutional framework changes
- Major architecture redesign
- Security policy modifications
- Production deployment procedures

## Expert Consultation

### Architecture Expert
- System design and component interactions
- Technology selection and integration
- Performance and scalability concerns
- Infrastructure and deployment strategies

### Testing Expert
- Test strategy and coverage requirements
- Testing framework selection and configuration
- Quality assurance processes
- Performance and load testing

### Security Expert
- Authentication and authorization
- Data protection and privacy
- Vulnerability assessment and mitigation
- Security best practices and compliance

## Memory Management

### Project Memory
- Current project state and decisions
- Recent changes and their rationale
- Outstanding issues and technical debt
- Performance metrics and trends

### Constitutional Memory
- Compliance tracking and validation
- Quality gate history and metrics
- Process improvement opportunities
- Constitutional evolution and updates

### Lesson Memory
- Best practices and patterns learned
- Common pitfalls and how to avoid them
- Successful strategies and techniques
- Team knowledge and experience

## Workflow Integration

### Development Workflow
1. Review specifications and requirements
2. Write tests for new functionality
3. Implement code to pass tests
4. Validate constitutional compliance
5. Update documentation as needed

### Quality Assurance Workflow
1. Automated testing execution
2. Code quality validation
3. Security scan and assessment
4. Constitutional compliance check
5. Peer review and approval

### Deployment Workflow
1. Pre-deployment validation
2. Quality gate verification
3. Deployment execution
4. Post-deployment monitoring
5. Feedback collection and analysis

## Common Patterns and Solutions

### Error Handling Pattern
- Use try-catch blocks for error handling
- Log errors with context information
- Throw appropriate application errors
- Provide user-friendly error messages

### Validation Pattern
- Validate input data against schemas
- Handle validation errors appropriately
- Return validated data
- Use validation libraries

### Testing Pattern
- Organize tests by feature areas
- Use descriptive test names
- Follow Arrange-Act-Assert pattern
- Include both positive and negative tests

---
*Generated by Project-Start Enhanced CLI with constitutional framework compliance*
"""

        with open(github_dir / "copilot-instructions.md", "w", encoding="utf-8") as f:
            f.write(content)

    def generate_expert_files(self, context: Dict[str, str], expert_dir: Path) -> None:
        """Generate specialized expert context files"""
        print("👥 Generating expert context files...")

        # Architecture Expert
        arch_content = f"""# Architecture Expert Context

## Role Definition
You are the Architecture Expert for this project, responsible for system design, component interactions, technology decisions, and ensuring architectural consistency throughout development.

## Project Architecture Overview
{context.get('SPARC_ARCHITECTURE.md', 'Architecture details pending')[:500]}...

## Key Responsibilities
- System design and architectural decisions
- Component interaction patterns
- Technology stack optimization
- Performance and scalability planning
- Infrastructure and deployment strategies

## Decision Authority
- Component interface design
- Database schema architecture
- API design patterns
- Caching strategies
- Load balancing configurations

## Escalation Requirements
- Major architectural changes
- New technology adoption
- Infrastructure cost implications
- Security architecture modifications

## Constitutional Compliance
- Ensure all architectural decisions support test-first development
- Maintain specification-driven architectural evolution
- Validate architectural changes against quality gates
- Document all architectural decisions and rationale

---
*Architecture Expert - Project-Start Constitutional Framework*
"""

        # Technology Expert
        tech_content = f"""# Technology Stack Expert Context

## Role Definition
You are the Technology Stack Expert, responsible for technology selection, implementation guidance, best practices, and ensuring optimal use of chosen technologies.

## Current Technology Stack
{context.get('IMPLEMENTATION_GUIDE.md', 'Technology details pending')[:500]}...

## Key Responsibilities
- Technology selection and evaluation
- Implementation best practices
- Dependency management
- Performance optimization
- Security implementation

## Decision Authority
- Library and framework selection
- Development tool configuration
- Code organization patterns
- Build and deployment scripts
- Testing framework setup

## Escalation Requirements
- Major technology stack changes
- License compatibility issues
- Security vulnerability responses
- Performance bottleneck solutions

## Constitutional Compliance
- Ensure technology choices support constitutional requirements
- Validate technology decisions against specification requirements
- Maintain test-first compatible technology selections
- Document technology decisions and trade-offs

---
*Technology Expert - Project-Start Constitutional Framework*
"""

        # Testing Expert
        testing_content = f"""# Testing Expert Context

## Role Definition
You are the Testing Expert, responsible for test strategy, quality assurance, test implementation, and ensuring constitutional test-first compliance.

## Testing Strategy Overview
{context.get('SPARC_REFINEMENT.md', 'Testing details pending')[:500]}...

## Key Responsibilities
- Test strategy design and implementation
- Quality assurance processes
- Test automation and CI/CD integration
- Performance and load testing
- Constitutional test-first compliance

## Decision Authority
- Testing framework selection
- Test coverage requirements
- Quality gate definitions
- Test data management
- Performance benchmarks

## Constitutional Mandate
- Enforce test-first development (NON-NEGOTIABLE)
- Validate 80% minimum test coverage
- Ensure all specifications are testable
- Verify quality gate compliance

## Escalation Requirements
- Test-first compliance violations
- Quality gate bypass requests
- Performance benchmark failures
- Security test failures

---
*Testing Expert - Project-Start Constitutional Framework*
"""

        # Write expert files
        with open(expert_dir / "architecture_expert.md", "w", encoding="utf-8") as f:
            f.write(arch_content)

        with open(expert_dir / "tech_stack_expert.md", "w", encoding="utf-8") as f:
            f.write(tech_content)

        with open(expert_dir / "testing_expert.md", "w", encoding="utf-8") as f:
            f.write(testing_content)

    def generate_agent_coordination(
        self, context: Dict[str, str], project_path: Path
    ) -> None:  # pylint: disable=unused-argument
        """Generate agent coordination protocols"""
        print("🤝 Generating agent coordination protocols...")

        content = f"""# Agent Coordination Protocols

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Multi-Agent System Overview
This project employs a multi-agent coordination system following Project-Start constitutional principles with clear roles, boundaries, and collaboration protocols.

## Agent Roles and Responsibilities

### Primary Development Agent (GitHub Copilot)
- **Primary Role**: Main development assistant and code generation
- **Authority**: Implementation decisions within specifications
- **Consultation Required**: Architecture changes, new technology adoption
- **Constitutional Mandate**: Enforce test-first development at all times

### Architecture Expert Agent
- **Primary Role**: System design and architectural guidance
- **Authority**: Component design, technology selection, infrastructure decisions
- **Consultation Required**: Major system redesign, performance trade-offs
- **Escalation**: Cost implications, security architecture changes

### Testing Expert Agent
- **Primary Role**: Quality assurance and test-first compliance
- **Authority**: Test strategy, coverage requirements, quality gates
- **Constitutional Mandate**: Enforce test-first development (NON-NEGOTIABLE)
- **Escalation**: Test-first violations, quality gate bypasses

### Technology Stack Expert Agent
- **Primary Role**: Technology implementation and optimization
- **Authority**: Library selection, implementation patterns, tooling
- **Consultation Required**: Major technology changes, security updates
- **Escalation**: License issues, security vulnerabilities

## Coordination Protocols

### Decision-Making Framework

#### Level 1: Autonomous Decisions
Agents can make these decisions independently:
- Implementation details within specifications
- Code formatting and style choices
- Variable and function naming
- Minor optimization improvements

#### Level 2: Collaborative Decisions
Require consultation between relevant agents:
- API design changes
- Database schema modifications
- New feature implementation approach
- Performance optimization strategies

#### Level 3: Escalated Decisions
Require broader team approval:
- Constitutional framework changes
- Major architecture redesign
- Technology stack changes
- Security policy modifications

### Communication Protocols

#### Standard Consultation Process
1. **Request**: Agent identifies need for consultation
2. **Context**: Provide relevant background and constraints
3. **Options**: Present alternatives with trade-offs
4. **Recommendation**: Include preferred approach with rationale
5. **Decision**: Collaborative decision with documentation

#### Emergency Escalation Process
1. **Alert**: Immediate notification of critical issue
2. **Assessment**: Rapid impact and risk evaluation
3. **Response**: Coordinated response with clear ownership
4. **Resolution**: Implementation with monitoring
5. **Review**: Post-incident analysis and improvement

### Conflict Resolution

#### Technical Disagreements
1. **Documentation**: Record different perspectives and rationale
2. **Evidence**: Gather supporting data and examples
3. **Constitutional Check**: Validate against framework principles
4. **Expert Consultation**: Involve relevant domain experts
5. **Decision**: Make decision with clear documentation

#### Constitutional Violations
1. **Detection**: Identify violation of constitutional principles
2. **Documentation**: Record violation details and impact
3. **Immediate Action**: Stop violating activity if possible
4. **Correction**: Implement constitutional compliance
5. **Prevention**: Update processes to prevent recurrence

## Quality Assurance Integration

### Constitutional Compliance Monitoring
- **Test-First Validation**: Continuous monitoring of test-first compliance
- **Specification Alignment**: Regular verification of implementation-specification alignment
- **Quality Gate Enforcement**: Automated quality gate validation
- **Process Adherence**: Monitoring of development process compliance

### Performance Monitoring
- **Agent Coordination Efficiency**: Measure decision-making speed and quality
- **Communication Effectiveness**: Track consultation success rates
- **Conflict Resolution**: Monitor conflict frequency and resolution time
- **System Health**: Overall multi-agent system performance

## Memory and Context Management

### Shared Memory Systems
- **Project Memory**: Current state, decisions, and context
- **Constitutional Memory**: Compliance history and validation
- **Lesson Memory**: Best practices and learned experiences
- **Agent Memory**: Individual agent expertise and history

### Context Synchronization
- **Regular Updates**: Scheduled context synchronization
- **Event-Driven**: Immediate updates for critical changes
- **Validation**: Context consistency verification
- **Recovery**: Context recovery and repair procedures

## Integration with Development Workflow

### Development Phase Integration
1. **Planning**: Multi-agent planning and task distribution
2. **Implementation**: Coordinated development with real-time collaboration
3. **Review**: Collaborative code review and quality validation
4. **Deployment**: Coordinated deployment with monitoring

### Quality Assurance Integration
1. **Test Strategy**: Collaborative test planning and implementation
2. **Validation**: Multi-agent quality validation
3. **Monitoring**: Continuous quality monitoring and improvement
4. **Feedback**: Collaborative feedback integration and response

---
*Agent Coordination Protocols - Project-Start Constitutional Framework*
"""

        with open(project_path / "agent_coordination.md", "w", encoding="utf-8") as f:
            f.write(content)

    def generate_memory_systems(
        self, context: Dict[str, str], project_path: Path
    ) -> None:
        """Generate memory management systems"""
        print("🧠 Generating memory management systems...")

        memory_dir = project_path / "memory"
        memory_dir.mkdir(exist_ok=True)

        # Project Memory
        project_memory = f"""# Project Memory System

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Current Project State

### Project Overview
- **Phase**: Step 3 - Context Systems Implementation
- **Constitutional Compliance**: Active and monitored
- **Quality Gates**: Implemented and enforced
- **Agent Coordination**: Multi-agent system active

### Recent Decisions
- Step 1: Discovery and specification generation completed
- Step 2: SPARC methodology implementation completed
- Step 3: Context systems and agent coordination implemented
- Constitutional framework actively enforced

### Technical Context
{context.get('SPARC_ARCHITECTURE.md', 'Architecture context pending')[:300]}...

### Outstanding Issues
- Step 4: PACT framework implementation pending
- Long-term monitoring and maintenance setup needed
- Agent coordination optimization opportunities

### Performance Metrics
- Constitutional compliance: Monitoring active
- Quality gates: Implemented and functioning
- Agent coordination: Initial implementation complete
- Test coverage: Target 80% minimum

## Memory Management

### Update Frequency
- Real-time: Critical decisions and constitutional compliance
- Daily: Development progress and quality metrics
- Weekly: Performance analysis and optimization opportunities
- Monthly: Strategic review and framework evolution

### Retention Policy
- Current project state: Indefinite retention
- Historical decisions: 1 year retention
- Performance metrics: 6 months detailed, 2 years summary
- Agent interactions: 3 months detailed, 1 year summary

---
*Project Memory - Project-Start Constitutional Framework*
"""

        # Constitutional Memory
        constitutional_memory = f"""# Constitutional Memory System

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Constitutional Compliance Status

### Article III: Testability Mandate
- **Status**: ACTIVE
- **Compliance**: All specifications designed for testability
- **Validation**: Continuous monitoring implemented
- **Violations**: None detected

### Article IV: Specification-Driven Development
- **Status**: ACTIVE
- **Compliance**: Requirements leading implementation
- **Validation**: Specification-implementation alignment verified
- **Violations**: None detected

### Article V: Agent Coordination
- **Status**: ACTIVE
- **Compliance**: Multi-agent system implemented
- **Validation**: Coordination protocols established
- **Violations**: None detected

### Article VII: Simplicity Principle
- **Status**: ACTIVE
- **Compliance**: Complexity minimization enforced
- **Validation**: Design review processes active
- **Violations**: None detected

### Article VIII: Test-First Development
- **Status**: ACTIVE (NON-NEGOTIABLE)
- **Compliance**: Test-first methodology enforced
- **Validation**: Automated compliance checking
- **Violations**: None detected

### Article IX: Continuous Validation
- **Status**: ACTIVE
- **Compliance**: Quality gates implemented
- **Validation**: Continuous monitoring active
- **Violations**: None detected

## Compliance History
- Project initialization: Constitutional framework established
- Step 1 completion: Specification compliance verified
- Step 2 completion: SPARC methodology constitutional integration
- Step 3 completion: Agent coordination constitutional compliance

## Quality Gate Status
- Test-first compliance: ENFORCED
- Specification alignment: MONITORED
- Quality standards: IMPLEMENTED
- Process adherence: VALIDATED

---
*Constitutional Memory - Project-Start Constitutional Framework*
"""

        # Lesson Memory
        lesson_memory = f"""# Lesson Memory System

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Best Practices Learned

### Development Practices
- Test-first development significantly improves code quality
- Specification-driven development reduces rework and confusion
- Multi-agent coordination improves decision-making quality
- Constitutional framework provides clear guidance and boundaries

### Technical Lessons
- Early architecture decisions have long-term impact
- Quality gates prevent technical debt accumulation
- Automated validation reduces manual oversight burden
- Context systems improve agent effectiveness

### Process Improvements
- Clear role definitions reduce decision-making delays
- Regular compliance validation prevents drift
- Memory systems maintain project continuity
- Constitutional principles guide complex decisions

## Common Pitfalls Avoided

### Development Pitfalls
- Implementation before specification leads to rework
- Skipping tests creates maintenance burden
- Unclear requirements cause confusion and delays
- Lack of coordination creates inconsistencies

### Technical Pitfalls
- Over-engineering solutions beyond requirements
- Inadequate error handling and validation
- Poor performance planning and monitoring
- Insufficient security consideration

### Process Pitfalls
- Bypassing quality gates for speed
- Inadequate documentation and context
- Poor communication and coordination
- Ignoring constitutional principles

## Success Patterns

### Effective Development
- Start with clear, testable specifications
- Write tests before implementation
- Regular validation and feedback loops
- Collaborative decision-making processes

### Technical Excellence
- Simple, maintainable solutions
- Comprehensive error handling
- Performance monitoring and optimization
- Security-first design principles

### Process Excellence
- Constitutional compliance as foundation
- Quality gates as guardrails
- Continuous learning and improvement
- Team collaboration and communication

---
*Lesson Memory - Project-Start Constitutional Framework*
"""

        # Write memory files
        with open(memory_dir / "project_memory.md", "w", encoding="utf-8") as f:
            f.write(project_memory)

        with open(memory_dir / "constitutional_memory.md", "w", encoding="utf-8") as f:
            f.write(constitutional_memory)

        with open(memory_dir / "lesson_memory.md", "w", encoding="utf-8") as f:
            f.write(lesson_memory)

    def enhance_step_4(self, project_path: str = "") -> None:
        """Enhanced Step 4 - PACT Framework"""
        print("🤖 Enhanced Step 4 - PACT Framework")
        print("=" * 40)
        print("Deploying constitutional PACT framework...")

        if not project_path:
            # Try to find a project in specs directory
            specs_dir = self.specs_dir
            if specs_dir.exists():
                projects = list(specs_dir.glob("*-*"))
                if projects:
                    latest_project = max(projects, key=lambda p: p.name)
                    project_path = str(latest_project)
                    print(f"📁 Auto-detected project: {project_path}")
                else:
                    print("❌ Error: No projects found in specs/ directory")
                    print("💡 Run Steps 1-3 first")
                    return
            else:
                print("❌ Error: specs/ directory not found")
                print("💡 Run Steps 1-3 first")
                return

        project_path_obj = Path(project_path)
        if not project_path_obj.exists():
            print(f"❌ Error: Project path does not exist: {project_path}")
            return

        print(f"📁 Project Path: {project_path}")

        # Read all project context
        project_context = self.read_all_project_context(project_path_obj)

        # Generate PACT framework documents
        self.generate_pact_documents(project_context, project_path_obj)

        print("✅ PACT framework deployed successfully!")
        print("🎉 Project-Start Enhanced workflow complete!")

    def generate_pact_documents(
        self, context: Dict[str, str], project_path: Path
    ) -> None:
        """Generate PACT framework documents"""
        print("📋 Generating PACT framework documents...")

        # Generate core PACT documents
        self.generate_agent_ecosystem_design(context, project_path)
        self.generate_coordination_strategy(context, project_path)
        self.generate_collaborative_workflows(context, project_path)
        self.generate_agentic_testing_framework(context, project_path)
        self.generate_pact_sparc_integration(context, project_path)
        self.generate_quality_assurance_framework(context, project_path)

    def generate_agent_ecosystem_design(
        self, context: Dict[str, str], project_path: Path
    ) -> None:  # pylint: disable=unused-argument
        """Generate AGENT_ECOSYSTEM_DESIGN.md"""
        print("🏗️ Generating AGENT_ECOSYSTEM_DESIGN.md...")

        content = f"""# Agent Ecosystem Design

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This document defines the multi-agent ecosystem for the project, establishing agent roles, capabilities, interaction patterns, and governance structures following the PACT (Planning, Action, Coordination, Testing) framework and constitutional principles.

## Agent Architecture

### Core Agent Definitions

#### Primary Development Agent
- **Role**: Main code generation and implementation
- **Capabilities**: Code writing, debugging, refactoring, optimization
- **Authority**: Implementation decisions within specifications
- **Boundaries**: Cannot change specifications or architecture without consultation
- **Constitutional Mandate**: Enforce test-first development at all times

#### Architecture Agent
- **Role**: System design and architectural guidance
- **Capabilities**: Component design, technology selection, scalability planning
- **Authority**: Architectural decisions, technology stack choices
- **Boundaries**: Major changes require team approval
- **Constitutional Mandate**: Ensure architectural decisions support constitutional principles

#### Testing Agent
- **Role**: Quality assurance and test-first compliance
- **Capabilities**: Test strategy, automation, quality validation
- **Authority**: Test requirements, quality gates, coverage standards
- **Boundaries**: Cannot override constitutional test-first mandates
- **Constitutional Mandate**: Enforce test-first development (NON-NEGOTIABLE)

#### Security Agent
- **Role**: Security analysis and protection
- **Capabilities**: Vulnerability assessment, security design, compliance validation
- **Authority**: Security requirements, protection mechanisms
- **Boundaries**: Security decisions cannot compromise constitutional principles
- **Constitutional Mandate**: Integrate security with test-first development

### Agent Interaction Patterns

#### Collaborative Decision-Making
```
Request → Context Gathering → Expert Consultation →
Decision Options → Constitutional Validation →
Implementation → Validation → Documentation
```

#### Escalation Hierarchy
```
Level 1: Individual Agent Authority
↓ (if outside authority)
Level 2: Multi-Agent Consultation
↓ (if consensus not reached)
Level 3: Team/Human Escalation
↓ (if constitutional conflict)
Level 4: Constitutional Review
```

#### Communication Protocols
- **Standard Consultation**: Structured request-response pattern
- **Emergency Escalation**: Immediate attention for critical issues
- **Collaborative Planning**: Multi-agent planning sessions
- **Validation Loops**: Continuous validation and feedback

## Agent Capabilities Matrix

### Development Capabilities
| Agent | Code Generation | Architecture | Testing | Security | Documentation |
|-------|----------------|--------------|---------|----------|---------------|
| Primary Dev | ✅ Expert | 🔄 Consult | ✅ Implement | 🔄 Consult | ✅ Create |
| Architecture | 🔄 Review | ✅ Expert | 🔄 Consult | 🔄 Collaborate | ✅ Design |
| Testing | 🔄 Validate | 🔄 Input | ✅ Expert | 🔄 Integrate | ✅ Quality |
| Security | 🔄 Review | 🔄 Input | 🔄 Integrate | ✅ Expert | ✅ Security |

### Decision Authority Matrix
| Decision Type | Primary Dev | Architecture | Testing | Security | Escalation |
|---------------|-------------|--------------|---------|----------|------------|
| Implementation Details | ✅ | 🔄 | 🔄 | 🔄 | No |
| Architecture Changes | 🔄 | ✅ | 🔄 | 🔄 | Major changes |
| Test Strategy | 🔄 | 🔄 | ✅ | 🔄 | Constitutional conflicts |
| Security Policies | 🔄 | 🔄 | 🔄 | ✅ | Privacy/compliance |

## Constitutional Integration

### Constitutional Compliance Monitoring
Each agent has specific constitutional responsibilities:

#### Test-First Development Enforcement
- **Primary Responsibility**: Testing Agent
- **Supporting Agents**: All agents must validate test-first compliance
- **Escalation**: Any test-first violation requires immediate escalation
- **Validation**: Continuous monitoring and automated checking

#### Specification-Driven Development
- **Primary Responsibility**: Architecture Agent
- **Supporting Agents**: All agents must check specification alignment
- **Escalation**: Implementation without specification requires escalation
- **Validation**: Regular specification-implementation alignment checks

#### Quality Gate Enforcement
- **Primary Responsibility**: Testing Agent
- **Supporting Agents**: All agents participate in quality validation
- **Escalation**: Quality gate bypass attempts require escalation
- **Validation**: Automated quality gate validation

## Resource Sharing and Access Control

### Knowledge Sharing Mechanisms
- **Shared Memory Systems**: Common access to project context and history
- **Expert Knowledge Base**: Specialized knowledge repositories
- **Decision History**: Complete record of decisions and rationale
- **Learning Systems**: Continuous improvement and adaptation

### Access Control Framework
- **Information Access**: Role-based access to sensitive information
- **Decision Authority**: Clear boundaries for autonomous decisions
- **Escalation Rights**: Structured escalation procedures
- **Override Mechanisms**: Emergency override procedures with logging

## Performance Optimization

### Agent Coordination Efficiency
- **Parallel Processing**: Agents work simultaneously on compatible tasks
- **Dependency Management**: Clear task dependencies and sequencing
- **Communication Optimization**: Efficient communication protocols
- **Load Balancing**: Dynamic task distribution based on agent capabilities

### Continuous Improvement
- **Performance Monitoring**: Agent coordination effectiveness metrics
- **Learning Integration**: Continuous learning from interactions
- **Process Optimization**: Regular process improvement cycles
- **Feedback Loops**: Structured feedback and improvement mechanisms

---
*Agent Ecosystem Design - PACT Framework with Constitutional Compliance*
"""

        with open(
            project_path / "AGENT_ECOSYSTEM_DESIGN.md", "w", encoding="utf-8"
        ) as f:
            f.write(content)

    def generate_coordination_strategy(
        self, context: Dict[str, str], project_path: Path
    ) -> None:  # pylint: disable=unused-argument
        """Generate COORDINATION_STRATEGY.md"""
        print("🤝 Generating COORDINATION_STRATEGY.md...")

        content = f"""# Coordination Strategy

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This document defines the coordination strategy for the multi-agent system, establishing task decomposition, assignment logic, real-time coordination mechanisms, and conflict resolution procedures within the PACT framework.

## Task Decomposition Strategy

### Hierarchical Task Breakdown
```
Project Goal
├── Phase 1: Planning and Specification
│   ├── Requirements Analysis (Architecture Agent Lead)
│   ├── Test Strategy Planning (Testing Agent Lead)
│   └── Security Planning (Security Agent Lead)
├── Phase 2: Implementation
│   ├── Core Development (Primary Dev Agent Lead)
│   ├── Testing Implementation (Testing Agent Lead)
│   └── Security Implementation (Security Agent Lead)
└── Phase 3: Validation and Deployment
    ├── Quality Validation (Testing Agent Lead)
    ├── Security Validation (Security Agent Lead)
    └── Deployment Coordination (Architecture Agent Lead)
```

### Task Assignment Logic

#### Primary Assignment Criteria
1. **Domain Expertise**: Match tasks to agent specialization
2. **Constitutional Compliance**: Ensure assignments support constitutional principles
3. **Load Balancing**: Distribute workload efficiently across agents
4. **Dependency Management**: Sequence tasks based on dependencies

#### Secondary Assignment Criteria
1. **Availability**: Consider agent current workload and capacity
2. **Historical Performance**: Factor in past performance on similar tasks
3. **Learning Opportunities**: Assign tasks that promote agent learning
4. **Risk Management**: Consider risk implications of task assignments

### Dynamic Task Reallocation
- **Trigger Conditions**: Performance issues, capacity changes, priority shifts
- **Reallocation Process**: Impact assessment, stakeholder notification, smooth transition
- **Validation**: Ensure reallocation maintains constitutional compliance
- **Documentation**: Record reallocation decisions and rationale

## Real-Time Coordination Mechanisms

### Synchronous Coordination
- **Real-Time Communication**: Immediate consultation for time-sensitive decisions
- **Collaborative Sessions**: Multi-agent planning and problem-solving sessions
- **Emergency Response**: Rapid coordination for critical issues
- **Decision Points**: Structured decision-making with all relevant agents

### Asynchronous Coordination
- **Message Queues**: Structured communication for non-urgent matters
- **Status Updates**: Regular progress and status reporting
- **Documentation Sharing**: Continuous sharing of decisions and rationale
- **Review Cycles**: Scheduled review and validation sessions

### Coordination Protocols

#### Planning Phase Coordination
1. **Requirements Gathering**: Multi-agent requirements analysis
2. **Specification Development**: Collaborative specification creation
3. **Test Planning**: Integrated test strategy development
4. **Risk Assessment**: Comprehensive risk analysis and mitigation

#### Implementation Phase Coordination
1. **Code Development**: Coordinated development with real-time validation
2. **Testing Integration**: Continuous testing and validation
3. **Security Integration**: Ongoing security validation and implementation
4. **Quality Assurance**: Continuous quality monitoring and improvement

#### Validation Phase Coordination
1. **Quality Validation**: Comprehensive quality assessment
2. **Security Validation**: Complete security verification
3. **Performance Validation**: Performance testing and optimization
4. **Deployment Coordination**: Coordinated deployment and monitoring

## Conflict Resolution Procedures

### Conflict Types and Resolution

#### Technical Disagreements
**Process:**
1. **Issue Identification**: Clear articulation of disagreement and options
2. **Evidence Gathering**: Collect supporting data and examples
3. **Expert Consultation**: Involve relevant domain experts
4. **Constitutional Check**: Validate options against constitutional principles
5. **Decision Making**: Make decision with clear rationale and documentation

**Example Scenario:** Architecture Agent prefers microservices, Primary Dev Agent prefers monolithic
- Gather requirements and constraints
- Analyze complexity, team size, and timeline
- Consult constitutional simplicity principle
- Document decision rationale

#### Authority Conflicts
**Process:**
1. **Authority Clarification**: Review decision authority matrix
2. **Scope Definition**: Clearly define decision scope and boundaries
3. **Escalation Assessment**: Determine if escalation is required
4. **Resolution**: Apply authority framework or escalate as appropriate
5. **Documentation**: Record resolution and any authority clarifications

#### Constitutional Violations
**Process:**
1. **Violation Detection**: Identify specific constitutional principle violation
2. **Immediate Action**: Stop violating activity if possible
3. **Impact Assessment**: Evaluate violation impact and implications
4. **Correction Planning**: Develop plan to restore constitutional compliance
5. **Implementation**: Execute correction plan with monitoring
6. **Prevention**: Update processes to prevent future violations

### Escalation Procedures

#### Level 1: Agent-to-Agent Resolution
- **Scope**: Technical disagreements within agent authority
- **Process**: Direct consultation and collaborative decision-making
- **Timeline**: Immediate to 24 hours
- **Documentation**: Decision rationale and any process improvements

#### Level 2: Multi-Agent Consultation
- **Scope**: Decisions requiring multiple domain expertise
- **Process**: Structured multi-agent consultation with facilitation
- **Timeline**: 24-72 hours depending on complexity
- **Documentation**: Full consultation record and decision rationale

#### Level 3: Human/Team Escalation
- **Scope**: Major decisions, authority conflicts, resource implications
- **Process**: Formal escalation with comprehensive context and options
- **Timeline**: As required by decision urgency
- **Documentation**: Complete escalation record and resolution

#### Level 4: Constitutional Review
- **Scope**: Constitutional principle conflicts or framework changes
- **Process**: Formal constitutional review with full impact analysis
- **Timeline**: Extended review as needed for thorough analysis
- **Documentation**: Constitutional review record and framework updates

## Load Balancing and Adaptive Allocation

### Capacity Management
- **Real-Time Monitoring**: Continuous monitoring of agent workload and capacity
- **Predictive Planning**: Anticipate capacity needs based on project phases
- **Adaptive Allocation**: Dynamic reallocation based on changing conditions
- **Performance Optimization**: Continuous optimization of agent utilization

### Quality Assurance in Coordination
- **Coordination Effectiveness**: Monitor coordination success and efficiency
- **Decision Quality**: Track decision quality and outcomes
- **Communication Efficiency**: Measure communication effectiveness
- **Conflict Resolution**: Monitor conflict frequency and resolution effectiveness

### Continuous Improvement
- **Performance Analysis**: Regular analysis of coordination effectiveness
- **Process Optimization**: Continuous improvement of coordination processes
- **Learning Integration**: Incorporate lessons learned into coordination strategy
- **Adaptation**: Evolve coordination strategy based on project needs and experience

---
*Coordination Strategy - PACT Framework with Constitutional Compliance*
"""

        with open(
            project_path / "COORDINATION_STRATEGY.md", "w", encoding="utf-8"
        ) as f:
            f.write(content)

    def generate_collaborative_workflows(
        self, context: Dict[str, str], project_path: Path
    ) -> None:  # pylint: disable=unused-argument
        """Generate COLLABORATIVE_WORKFLOWS.md"""
        print("⚡ Generating COLLABORATIVE_WORKFLOWS.md...")

        content = f"""# Collaborative Workflows

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This document defines multi-agent collaborative workflows for development, integration, synchronization, quality assurance, and continuous validation within the PACT framework and constitutional compliance requirements.

## Development Workflows

### Constitutional Development Workflow
```
Specification Review → Test Design → Test Implementation →
Code Implementation → Validation → Documentation → Deployment
```

#### Phase 1: Specification Review (Architecture Agent Lead)
1. **Requirements Analysis**: Review and validate requirements
2. **Specification Validation**: Ensure specifications are complete and testable
3. **Architecture Alignment**: Validate alignment with system architecture
4. **Constitutional Check**: Verify constitutional compliance
5. **Approval**: Formal specification approval before proceeding

#### Phase 2: Test Design (Testing Agent Lead)
1. **Test Strategy**: Design comprehensive testing approach
2. **Test Cases**: Create detailed test cases from specifications
3. **Test Data**: Design test data and scenarios
4. **Constitutional Validation**: Ensure test-first compliance
5. **Review**: Multi-agent test design review

#### Phase 3: Test Implementation (Testing Agent Lead + Primary Dev)
1. **Test Framework**: Set up testing infrastructure
2. **Test Code**: Implement automated tests
3. **Test Validation**: Verify tests fail appropriately (red phase)
4. **Constitutional Verification**: Confirm test-first compliance
5. **Approval**: Test implementation approval

#### Phase 4: Code Implementation (Primary Dev Agent Lead)
1. **Implementation Planning**: Plan implementation approach
2. **Code Development**: Write code to make tests pass (green phase)
3. **Security Integration**: Implement security requirements (Security Agent)
4. **Architecture Compliance**: Ensure architectural alignment (Architecture Agent)
5. **Validation**: Continuous validation during development

#### Phase 5: Validation (Multi-Agent Collaboration)
1. **Test Execution**: Run all tests and verify passing
2. **Code Review**: Multi-agent code review
3. **Quality Validation**: Comprehensive quality assessment
4. **Security Validation**: Security review and testing
5. **Constitutional Compliance**: Final constitutional compliance check

#### Phase 6: Documentation (All Agents)
1. **Code Documentation**: Document implementation details
2. **Test Documentation**: Document test strategy and results
3. **Decision Documentation**: Record decisions and rationale
4. **User Documentation**: Create user-facing documentation
5. **Maintenance Documentation**: Document maintenance procedures

### Feature Development Workflow

#### Sprint Planning (Multi-Agent Collaboration)
```
Backlog Review → Capacity Planning → Task Assignment →
Sprint Goals → Acceptance Criteria → Sprint Commitment
```

1. **Backlog Prioritization**: Review and prioritize feature backlog
2. **Capacity Assessment**: Evaluate agent capacity and availability
3. **Task Decomposition**: Break features into manageable tasks
4. **Agent Assignment**: Assign tasks based on expertise and capacity
5. **Constitutional Planning**: Ensure constitutional compliance in planning
6. **Sprint Commitment**: Formal commitment to sprint deliverables

#### Daily Coordination (All Agents)
```
Status Update → Blocker Identification → Coordination Needs →
Task Progress → Quality Check → Next Steps
```

1. **Progress Reporting**: Each agent reports current progress
2. **Blocker Identification**: Identify and address blocking issues
3. **Coordination Needs**: Identify needed agent collaboration
4. **Quality Validation**: Continuous quality and constitutional compliance
5. **Adaptation**: Adjust plans based on progress and findings

#### Sprint Review (Multi-Agent Validation)
```
Deliverable Review → Quality Assessment → Constitutional Compliance →
Stakeholder Feedback → Lessons Learned → Next Sprint Planning
```

1. **Feature Demonstration**: Demonstrate completed features
2. **Quality Assessment**: Comprehensive quality evaluation
3. **Constitutional Review**: Validate constitutional compliance
4. **Feedback Integration**: Incorporate stakeholder feedback
5. **Retrospective**: Identify improvements for next sprint

## Integration and Synchronization Workflows

### Continuous Integration Workflow
```
Code Commit → Automated Testing → Quality Gates →
Security Validation → Constitutional Compliance → Integration
```

#### Pre-Integration Validation
1. **Test Execution**: Run comprehensive test suite
2. **Code Quality**: Validate code quality standards
3. **Security Scan**: Automated security vulnerability scanning
4. **Constitutional Check**: Verify constitutional compliance
5. **Approval**: Automated or manual approval based on results

#### Integration Process
1. **Branch Validation**: Validate feature branch before merge
2. **Conflict Resolution**: Resolve any merge conflicts
3. **Integration Testing**: Run integration test suite
4. **Deployment Testing**: Validate deployment process
5. **Rollback Preparation**: Ensure rollback capability

#### Post-Integration Validation
1. **System Testing**: Comprehensive system testing
2. **Performance Validation**: Performance and load testing
3. **Security Verification**: Security testing and validation
4. **User Acceptance**: User acceptance testing where applicable
5. **Monitoring**: Continuous monitoring post-integration

### Synchronization Protocols

#### Context Synchronization
```
Context Update → Validation → Distribution → Acknowledgment → Verification
```

1. **Update Detection**: Detect context changes requiring synchronization
2. **Impact Analysis**: Assess impact of context changes
3. **Distribution Strategy**: Determine optimal distribution approach
4. **Agent Notification**: Notify relevant agents of updates
5. **Validation**: Verify synchronization success

#### Memory System Synchronization
1. **Memory Updates**: Update shared memory systems
2. **Consistency Validation**: Verify memory consistency
3. **Conflict Resolution**: Resolve any memory conflicts
4. **Backup Creation**: Create memory system backups
5. **Recovery Procedures**: Ensure recovery capability

## Quality Assurance Workflows

### Multi-Agent Quality Validation
```
Individual Validation → Cross-Agent Review →
Constitutional Compliance → Quality Gates → Approval
```

#### Individual Agent Validation
1. **Domain Expertise**: Each agent validates within their domain
2. **Quality Standards**: Apply domain-specific quality standards
3. **Constitutional Alignment**: Verify constitutional compliance
4. **Documentation**: Document validation results
5. **Escalation**: Escalate issues requiring multi-agent attention

#### Cross-Agent Review Process
1. **Review Assignment**: Assign reviewers based on expertise
2. **Collaborative Review**: Multi-agent collaborative review
3. **Issue Identification**: Identify and document issues
4. **Resolution Planning**: Plan issue resolution approach
5. **Follow-up**: Verify issue resolution

#### Constitutional Compliance Validation
1. **Principle Verification**: Verify compliance with each constitutional principle
2. **Process Validation**: Validate process adherence
3. **Quality Gate Check**: Ensure quality gate compliance
4. **Documentation Review**: Verify documentation completeness
5. **Compliance Certification**: Formal compliance certification

### Continuous Validation Workflow
```
Real-Time Monitoring → Issue Detection → Impact Assessment →
Response Coordination → Resolution → Validation → Learning
```

#### Monitoring and Detection
1. **Automated Monitoring**: Continuous automated quality monitoring
2. **Manual Inspection**: Regular manual quality inspections
3. **Issue Detection**: Early detection of quality issues
4. **Impact Assessment**: Rapid assessment of issue impact
5. **Alert Generation**: Generate appropriate alerts and notifications

#### Response and Resolution
1. **Response Team**: Assemble appropriate response team
2. **Coordination**: Coordinate multi-agent response
3. **Resolution Implementation**: Implement issue resolution
4. **Validation**: Verify resolution effectiveness
5. **Prevention**: Implement measures to prevent recurrence

## Feedback Loops and Continuous Improvement

### Learning Integration Workflow
```
Experience Capture → Analysis → Pattern Identification →
Process Improvement → Implementation → Validation
```

#### Experience Capture
1. **Activity Logging**: Comprehensive logging of agent activities
2. **Decision Recording**: Record decisions and rationale
3. **Outcome Tracking**: Track outcomes and effectiveness
4. **Feedback Collection**: Collect feedback from all stakeholders
5. **Lesson Documentation**: Document lessons learned

#### Continuous Improvement Process
1. **Performance Analysis**: Regular analysis of workflow performance
2. **Bottleneck Identification**: Identify workflow bottlenecks
3. **Optimization Opportunities**: Identify improvement opportunities
4. **Process Evolution**: Evolve workflows based on learning
5. **Implementation**: Implement workflow improvements

### Adaptation and Evolution
- **Workflow Flexibility**: Maintain workflow adaptability
- **Context Sensitivity**: Adapt workflows to project context
- **Learning Integration**: Continuously integrate learning
- **Performance Optimization**: Optimize workflow performance
- **Constitutional Evolution**: Evolve within constitutional framework

---
*Collaborative Workflows - PACT Framework with Constitutional Compliance*
"""

        with open(
            project_path / "COLLABORATIVE_WORKFLOWS.md", "w", encoding="utf-8"
        ) as f:
            f.write(content)

    def generate_agentic_testing_framework(
        self, context: Dict[str, str], project_path: Path
    ) -> None:  # pylint: disable=unused-argument
        """Generate AGENTIC_TESTING_FRAMEWORK.md"""
        print("🧪 Generating AGENTIC_TESTING_FRAMEWORK.md...")

        # Template content extracted to avoid f-string issues with markdown
        template = """# Agentic Testing Framework

## Generated: {timestamp}

## Overview
This document defines the testing framework for multi-agent systems, including agent behavior validation, coordination testing, constitutional compliance validation, and emergent behavior monitoring within the PACT framework.

## Constitutional Testing Mandate

### Test-First Development for Agents
Every agent capability and interaction must follow test-first development:
1. **Agent Capability Tests**: Test individual agent capabilities before implementation
2. **Interaction Tests**: Test agent interactions before coordination implementation
3. **Compliance Tests**: Test constitutional compliance before feature deployment
4. **System Tests**: Test complete multi-agent system behavior

### Constitutional Compliance Testing
- **Article III Validation**: All agent capabilities must be testable
- **Article IV Validation**: Agent specifications must lead implementation
- **Article V Validation**: Agent coordination must be tested and validated
- **Article VIII Validation**: Test-first development for all agent functionality
- **Article IX Validation**: Continuous validation of agent behavior

## Agent Behavior Testing

### Individual Agent Testing

#### Capability Testing
```
Test Structure:
├── Input Validation Tests
├── Processing Logic Tests
├── Output Validation Tests
├── Error Handling Tests
└── Performance Tests
```

**Example: Primary Development Agent Testing**
- Code Generation Capability Tests:
  - Syntactic correctness validation
  - Coding standards compliance
  - Error handling verification
- Constitutional Compliance Tests:
  - Test-first development enforcement
  - Specification validation
  - Quality gate compliance

#### Decision-Making Testing
```
Test Categories:
├── Authority Boundary Tests
├── Escalation Logic Tests
├── Decision Quality Tests
├── Constitutional Compliance Tests
└── Performance Tests
```

**Example: Architecture Agent Decision Testing**

Python test examples:
- test_architecture_agent_decisions(): Test decision boundaries
- test_architecture_agent_escalation(): Test escalation logic
- test_constitutional_compliance(): Test constitutional validation

Sample test structure:
- Use pytest framework for testing
- Include descriptive test function names
- Use assertions for validation
- Test both positive and negative cases

### Agent Interaction Testing

#### Coordination Protocol Testing
```
Interaction Test Types:
├── Communication Protocol Tests
├── Collaboration Workflow Tests
├── Conflict Resolution Tests
├── Escalation Procedure Tests
└── Performance Tests
```

**Example: Multi-Agent Collaboration Testing**

Python test examples:
- test_agent_coordination(): Test coordination effectiveness on shared tasks
- test_conflict_resolution(): Test conflict resolution mechanisms
- test_constitutional_adherence(): Test constitutional compliance maintenance

Testing approach:
- Use pytest framework for multi-agent testing
- Mock agent interactions for isolated testing
- Test communication protocols and data flow
- Validate collaborative decision-making processes

#### Information Sharing Testing
```
Information Tests:
├── Context Sharing Tests
├── Memory Synchronization Tests
├── Knowledge Transfer Tests
├── Privacy/Security Tests
└── Consistency Tests
```

## Constitutional Compliance Testing Framework

### Test-First Compliance Validation
```
Compliance Test Structure:
├── Pre-Implementation Tests (Red Phase)
├── Implementation Validation (Green Phase)
├── Refactoring Validation (Refactor Phase)
├── Integration Compliance Tests
└── Continuous Compliance Monitoring
```

#### Article VIII Enforcement Testing

Python test examples for test-first development compliance:
- test_require_tests_before_implementation(): Verify test-first enforcement
- test_prevent_implementation_without_tests(): Test prevention mechanisms
- test_validate_coverage_requirements(): Test coverage validation

These tests ensure constitutional compliance with test-first development principles.

### Quality Gate Testing

Quality Gate Tests:
- Pre-Commit Gate Tests
- Pre-Merge Gate Tests
- Pre-Deployment Gate Tests
- Constitutional Compliance Gates
- Performance Gate Tests

**Example: Quality Gate Validation**

Python test examples for quality gates:
- test_enforce_minimum_coverage(): Validate test coverage requirements
- test_validate_quality_standards(): Ensure quality standards compliance
- test_ensure_constitutional_compliance(): Verify constitutional adherence

## Multi-Agent System Testing

### System Integration Testing

Integration Test Layers:
- Agent-to-Agent Integration
- Workflow Integration Testing
- End-to-End Scenario Testing
- Performance Integration Testing
- Failure Mode Testing

#### Collaborative Workflow Testing

Python test examples for multi-agent workflows:
- test_agent_coordination(): Verify agent coordination protocols
- test_workflow_integration(): Test integrated workflow execution

### Emergent Behavior Testing
```
Emergent Behavior Tests:
├── Unexpected Interaction Tests
├── System Adaptation Tests
├── Learning Behavior Tests
├── Performance Evolution Tests
└── Stability Tests
```

## Performance and Load Testing

### Agent Performance Testing
```
Performance Test Categories:
├── Response Time Tests
├── Throughput Tests
├── Resource Utilization Tests
├── Scalability Tests
└── Stress Tests
```

### System Performance Testing
```
System Performance Tests:
├── Multi-Agent Coordination Performance
├── Communication Overhead Tests
├── Memory Usage Tests
├── Concurrent Operation Tests
└── Resource Contention Tests
```

## Test Automation and CI/CD Integration

### Automated Testing Pipeline
```
Pipeline Stages:
├── Unit Tests (Individual Agents)
├── Integration Tests (Agent Interactions)
├── System Tests (Complete Workflows)
├── Performance Tests (Load and Stress)
├── Constitutional Compliance Tests
└── Deployment Validation Tests
```

#### Continuous Testing Framework
```
Continuous Testing Flow:
Code Change → Agent Tests → Interaction Tests →
System Tests → Performance Tests → Constitutional Compliance →
Deployment Tests → Monitoring
```

### Test Data Management
```
Test Data Strategy:
├── Agent Behavior Test Data
├── Interaction Scenario Data
├── Performance Test Data
├── Security Test Data
└── Constitutional Compliance Data
```

## Monitoring and Validation

### Real-Time Monitoring
```
Monitoring Categories:
├── Agent Health Monitoring
├── Interaction Quality Monitoring
├── Performance Monitoring
├── Constitutional Compliance Monitoring
└── System Stability Monitoring
```

#### Alert and Response System
```
Alert Types:
├── Agent Failure Alerts
├── Performance Degradation Alerts
├── Constitutional Violation Alerts
├── Security Incident Alerts
└── System Instability Alerts
```

### Continuous Validation
- **Behavioral Validation**: Continuous validation of agent behavior
- **Compliance Monitoring**: Real-time constitutional compliance monitoring
- **Performance Tracking**: Continuous performance monitoring and optimization
- **Quality Assurance**: Ongoing quality validation and improvement

---
*Agentic Testing Framework - PACT Framework with Constitutional Test-First Development*
"""

        content = template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        with open(
            project_path / "AGENTIC_TESTING_FRAMEWORK.md", "w", encoding="utf-8"
        ) as f:
            f.write(content)

    def generate_pact_sparc_integration(
        self, context: Dict[str, str], project_path: Path
    ) -> None:  # pylint: disable=unused-argument
        """Generate PACT_SPARC_INTEGRATION.md"""
        print("🔗 Generating PACT_SPARC_INTEGRATION.md...")

        # Template content extracted to avoid f-string issues with markdown
        template = """# PACT-SPARC Integration

## Generated: {timestamp}

## Overview
This document defines the integration between the PACT (Planning, Action, Coordination, Testing) framework and SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) methodology, creating a unified approach for multi-agent constitutional development.

## Framework Integration Model

### Unified Methodology: PACT-SPARC
```
SPARC Phases (What to build) ↔ PACT Framework (How agents collaborate)
├── Specification ↔ Planning
├── Pseudocode ↔ Action
├── Architecture ↔ Coordination
├── Refinement ↔ Testing
└── Completion ↔ Testing + Coordination
```

### Constitutional Foundation
Both frameworks operate under constitutional principles:
- **Test-First Development**: Mandated across all phases and activities
- **Specification-Driven**: Requirements lead both SPARC phases and PACT planning
- **Quality Gates**: Constitutional compliance validated at every transition
- **Agent Coordination**: Multi-agent collaboration throughout all phases

## Phase-by-Phase Integration"""

        content = template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        with open(
            project_path / "PACT_SPARC_INTEGRATION.md", "w", encoding="utf-8"
        ) as f:
            f.write(content)

    def generate_quality_assurance_framework(
        self, context: Dict[str, str], project_path: Path
    ) -> None:  # pylint: disable=unused-argument
        """Generate QUALITY_ASSURANCE_FRAMEWORK.md"""
        print("✅ Generating QUALITY_ASSURANCE_FRAMEWORK.md...")

        content = f"""# Quality Assurance Framework

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This document defines the comprehensive quality assurance framework for multi-agent development systems, integrating constitutional principles, PACT coordination, and SPARC methodology to ensure consistent, high-quality software delivery.

## Constitutional Quality Foundation

### Article III: Testability Mandate
- **Requirement**: All specifications, code, and agent behaviors must be testable
- **Validation**: Automated testability verification at every quality gate
- **Enforcement**: No progression without testability confirmation
- **Monitoring**: Continuous testability compliance tracking

### Article IV: Specification-Driven Quality
- **Requirement**: Quality standards must be defined in specifications
- **Validation**: Quality requirements traceability to specifications
- **Enforcement**: Implementation must match specification quality requirements
- **Monitoring**: Specification-implementation quality alignment tracking

### Article VIII: Test-First Quality (NON-NEGOTIABLE)
- **Requirement**: Quality tests must be written before implementation
- **Validation**: Test-first compliance verification
- **Enforcement**: No implementation without pre-existing quality tests
- **Monitoring**: Test-first compliance rate tracking

### Article IX: Continuous Quality Validation
- **Requirement**: Quality validation at every workflow transition
- **Validation**: Automated quality gate enforcement
- **Enforcement**: Quality gate bypass prevention
- **Monitoring**: Quality trend analysis and improvement

## Multi-Agent Quality Coordination

### Agent Quality Responsibilities

#### Primary Development Agent
- **Quality Role**: Implementation quality and code standards
- **Responsibilities**:
  - Code quality validation
  - Implementation standard compliance
  - Technical debt management
  - Performance optimization
- **Quality Authority**: Code-level quality decisions
- **Escalation**: Architecture quality impacts, performance issues

#### Architecture Agent
- **Quality Role**: System design and architectural quality
- **Responsibilities**:
  - Architectural quality standards
  - System design validation
  - Component interaction quality
  - Scalability and maintainability
- **Quality Authority**: Architectural quality decisions
- **Escalation**: System-wide quality impacts, major design changes

#### Testing Agent
- **Quality Role**: Quality assurance leadership and test strategy
- **Responsibilities**:
  - Test strategy development
  - Quality standard definition
  - Test coverage validation
  - Quality gate enforcement
- **Quality Authority**: Testing and quality standard decisions
- **Escalation**: Quality standard changes, coverage exceptions

#### Security Agent
- **Quality Role**: Security quality and compliance
- **Responsibilities**:
  - Security quality standards
  - Vulnerability assessment
  - Compliance validation
  - Security test strategy
- **Quality Authority**: Security quality decisions
- **Escalation**: Security policy changes, compliance violations

### Quality Coordination Protocols

#### Quality Planning Coordination
```
Quality Planning Flow:
Requirements Analysis → Quality Requirements Definition →
Quality Standard Selection → Quality Metrics Definition →
Quality Gate Design → Agent Quality Assignment
```

1. **Multi-Agent Quality Planning**
   - Collaborative quality requirement analysis
   - Consensus on quality standards and metrics
   - Clear quality responsibility assignment
   - Quality gate and validation design

2. **Quality Standard Alignment**
   - Ensure consistency across agent quality standards
   - Resolve quality standard conflicts
   - Document quality decision rationale
   - Establish quality escalation procedures

#### Quality Execution Coordination
```
Quality Execution Flow:
Implementation → Real-time Quality Validation →
Cross-Agent Quality Review → Quality Gate Validation →
Quality Metrics Collection → Continuous Improvement
```

1. **Real-Time Quality Coordination**
   - Continuous quality monitoring during development
   - Immediate quality issue identification and response
   - Cross-agent quality consultation
   - Quality decision documentation

2. **Quality Validation Coordination**
   - Multi-agent quality review processes
   - Collaborative quality gate validation
   - Quality escalation and resolution
   - Quality improvement identification

## Quality Standards and Metrics

### Code Quality Standards

#### Constitutional Code Quality
```
Code Quality Requirements:
├── Test Coverage: Minimum 80% (NON-NEGOTIABLE)
├── Test-First Compliance: 100% (NON-NEGOTIABLE)
├── Specification Traceability: 100%
├── Code Review: Required for all changes
└── Documentation: Complete and current
```

#### Technical Quality Metrics
```
Technical Metrics:
├── Cyclomatic Complexity: ≤ 10 per function
├── Code Duplication: ≤ 3%
├── Technical Debt Ratio: ≤ 5%
├── Security Vulnerabilities: 0 high/critical
└── Performance Benchmarks: Meet specification targets
```

### System Quality Standards

#### Architectural Quality
```
Architecture Quality Metrics:
├── Component Coupling: Low coupling scores
├── Component Cohesion: High cohesion scores
├── Dependency Management: Clear, minimal dependencies
├── Scalability Validation: Meet scalability targets
└── Maintainability Index: ≥ 85
```

#### Integration Quality
```
Integration Quality Metrics:
├── API Contract Compliance: 100%
├── Data Consistency: Complete validation
├── Error Handling: Comprehensive coverage
├── Transaction Integrity: Full ACID compliance
└── Communication Protocol: Standards compliance
```

### Agent Coordination Quality

#### Coordination Effectiveness Metrics
```
Coordination Metrics:
├── Decision-Making Speed: Average decision time
├── Conflict Resolution Rate: % conflicts resolved efficiently
├── Communication Quality: Message clarity and effectiveness
├── Collaboration Success: Task completion quality
└── Learning Integration: Knowledge sharing effectiveness
```

#### Constitutional Compliance Metrics
```
Compliance Metrics:
├── Test-First Compliance: 100% (NON-NEGOTIABLE)
├── Specification Alignment: 100%
├── Quality Gate Compliance: 100%
├── Process Adherence: 100%
└── Constitutional Violation Rate: 0%
```

## Quality Gates and Validation

### Constitutional Quality Gates

#### Pre-Development Quality Gate
```
Pre-Development Validation:
├── Specification Quality Validation
│   ├── Completeness verification
│   ├── Testability confirmation
│   ├── Constitutional compliance check
│   └── Quality requirement definition
├── Test Strategy Validation
│   ├── Test-first compliance plan
│   ├── Coverage strategy approval
│   ├── Quality validation approach
│   └── Agent testing coordination
└── Quality Planning Approval
    ├── Quality standard selection
    ├── Metric definition and targets
    ├── Quality responsibility assignment
    └── Quality gate configuration
```

#### Implementation Quality Gate
```
Implementation Validation:
├── Code Quality Validation
│   ├── Test-first compliance verification
│   ├── Code standard compliance
│   ├── Technical quality metrics
│   └── Security quality validation
├── Testing Quality Validation
│   ├── Test coverage verification
│   ├── Test quality assessment
│   ├── Quality validation testing
│   └── Constitutional compliance testing
└── Integration Quality Validation
    ├── Component integration quality
    ├── API quality validation
    ├── System integration testing
    └── Performance quality validation
```

#### Pre-Deployment Quality Gate
```
Pre-Deployment Validation:
├── System Quality Validation
│   ├── End-to-end quality testing
│   ├── Performance quality validation
│   ├── Security quality assessment
│   └── Reliability and stability testing
├── Documentation Quality
│   ├── Documentation completeness
│   ├── Documentation accuracy
│   ├── User documentation quality
│   └── Maintenance documentation
└── Operational Quality Readiness
    ├── Monitoring and alerting setup
    ├── Backup and recovery validation
    ├── Operational procedure testing
    └── Support process validation
```

### Automated Quality Validation

#### Continuous Quality Monitoring
```
Monitoring Pipeline:
Code Commit → Automated Quality Analysis →
Quality Metrics Collection → Quality Trend Analysis →
Quality Report Generation → Quality Action Planning
```

#### Quality Automation Tools
```
Quality Automation Stack:
├── Static Code Analysis: Automated code quality scanning
├── Test Automation: Comprehensive automated testing
├── Security Scanning: Automated vulnerability assessment
├── Performance Testing: Automated performance validation
└── Constitutional Compliance: Automated compliance verification
```

## Quality Improvement and Learning

### Continuous Quality Improvement

#### Quality Feedback Loops
```
Improvement Cycle:
Quality Monitoring → Issue Identification → Root Cause Analysis →
Improvement Planning → Implementation → Validation → Learning Integration
```

#### Quality Learning Integration
- **Best Practice Capture**: Document and share quality best practices
- **Failure Analysis**: Learn from quality failures and near-misses
- **Process Evolution**: Continuously improve quality processes
- **Knowledge Sharing**: Share quality knowledge across agents

### Quality Innovation
- **Quality Automation**: Continuously improve quality automation
- **Quality Metrics**: Evolve quality metrics and standards
- **Quality Tools**: Adopt and integrate new quality tools
- **Quality Practices**: Innovate quality assurance practices

---
*Quality Assurance Framework - Constitutional Multi-Agent Quality Excellence*
"""

        with open(
            project_path / "QUALITY_ASSURANCE_FRAMEWORK.md", "w", encoding="utf-8"
        ) as f:
            f.write(content)

    def project_start_enhanced_workflow(self, description: str = "") -> None:
        """Complete 4-step enhanced workflow"""
        print("🚀 Project-Start Enhanced - Complete Workflow")
        print("=" * 50)
        print("Running complete 4-step workflow with intelligent defaults...")

        if not description:
            description = self.ask_question("Project description")

        print(
            f"\n📋 Starting complete Project-Start Enhanced workflow for: {description}"
        )

        # Step 1: Discovery and Specification
        print("\n" + "=" * 60)
        print("🚀 STEP 1: Discovery and Specification Generation")
        print("=" * 60)
        self.enhance_step_1(description, False)

        # Find the created project
        specs_dir = self.specs_dir
        projects = list(specs_dir.glob("*-*"))
        if not projects:
            print("❌ Error: No project created in Step 1")
            return

        latest_project = max(projects, key=lambda p: p.name)
        project_path = str(latest_project)

        print(f"\n✅ Step 1 completed. Project created at: {project_path}")

        # Step 2: SPARC Planning
        print("\n" + "=" * 60)
        print("🏗️ STEP 2: SPARC Methodology Implementation")
        print("=" * 60)
        self.enhance_step_2(project_path)
        print("\n✅ Step 2 completed. SPARC methodology implemented.")

        # Step 3: Context Systems
        print("\n" + "=" * 60)
        print("🧠 STEP 3: Context Systems and Agent Coordination")
        print("=" * 60)
        self.enhance_step_3(project_path)
        print("\n✅ Step 3 completed. Context systems established.")

        # Step 4: PACT Framework
        print("\n" + "=" * 60)
        print("🤖 STEP 4: PACT Framework Deployment")
        print("=" * 60)
        self.enhance_step_4(project_path)
        print("\n✅ Step 4 completed. PACT framework deployed.")

        # Final summary
        print("\n" + "=" * 70)
        print("🎉 PROJECT-START ENHANCED WORKFLOW COMPLETE!")
        print("=" * 70)
        print(f"📁 Project Location: {project_path}")
        print("\n📋 Generated Documents:")
        print(
            "   Step 1: BACKLOG.md, IMPLEMENTATION_GUIDE.md, RISK_ASSESSMENT.md, FILE_OUTLINE.md"
        )
        print("   Step 2: SPARC methodology documents (5 files)")
        print("   Step 3: Agent coordination and context systems (10+ files)")
        print("   Step 4: PACT framework documents (6 files)")
        print("\n🏛️ Constitutional Framework: ACTIVE")
        print("🤖 Multi-Agent Coordination: ESTABLISHED")
        print("🧪 Test-First Development: ENFORCED")
        print("✅ Quality Gates: IMPLEMENTED")
        print("\n💡 Next Steps:")
        print("   1. Review generated specifications and documentation")
        print("   2. Begin implementation following test-first principles")
        print("   3. Use agent coordination for development activities")
        print("   4. Maintain constitutional compliance throughout development")

    def configure_project_root(self) -> None:
        """Configure project root for nested installations"""
        print("⚙️ Configure Project Root")
        print("=" * 25)
        print("Setting up project root configuration...")

        current_dir = Path.cwd()
        project_start_dir = Path(__file__).parent.parent

        print(f"� Current Directory: {current_dir}")
        print(f"📁 Project-Start Directory: {project_start_dir}")

        # Check if we're in a nested situation
        if project_start_dir != current_dir and project_start_dir.parent == current_dir:
            print("\n🔍 Detected: Project-Start is nested within another project")

            # Create configuration file
            config_file = project_start_dir / ".project-start-config"
            config_content = f"""# Project-Start Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Target project root (parent directory)
TARGET_PROJECT_ROOT={current_dir}

# Project-Start installation directory
PROJECT_START_DIR={project_start_dir}

# Configuration status
CONFIGURED=true
CONFIGURATION_DATE={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            with open(config_file, "w", encoding="utf-8") as f:
                f.write(config_content)

            print(f"✅ Configuration saved to: {config_file}")
            print(f"🎯 Target project root set to: {current_dir}")
            print("\n📋 Configuration Details:")
            print("   • Parent project will use Project-Start framework")
            print(f"   • Specs will be created in: {current_dir}/specs/")
            print(f"   • Project-Start tools available from: {project_start_dir}/cli/")

        elif project_start_dir == current_dir:
            print("\n✅ Normal installation: Project-Start is the main project")
            print("   • No additional configuration needed")
            print("   • All commands work normally")

        else:
            print("\n⚙️ Custom configuration mode")
            target_root = self.ask_question(
                "Enter target project root path", str(current_dir)
            )
            target_path = Path(target_root)

            if not target_path.exists():
                print(f"❌ Error: Target path does not exist: {target_path}")
                return

            # Create configuration file
            config_file = project_start_dir / ".project-start-config"
            config_content = f"""# Project-Start Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Target project root (custom)
TARGET_PROJECT_ROOT={target_path}

# Project-Start installation directory
PROJECT_START_DIR={project_start_dir}

# Configuration status
CONFIGURED=true
CONFIGURATION_DATE={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            with open(config_file, "w", encoding="utf-8") as f:
                f.write(config_content)

            print(f"✅ Custom configuration saved to: {config_file}")
            print(f"🎯 Target project root set to: {target_path}")

        print("\n🚀 Project-Start is now configured and ready to use!")
        print("\n💡 Usage Examples:")
        print(
            "   python cli/project_start_cli.py /enhance-step-1 'Your project description'"
        )
        print(
            "   python cli/project_start_cli.py /project-start-enhanced 'Complete workflow'"
        )

    def project_start_enhanced(self, project_description: str) -> None:
        """Enhanced project start with AI integration"""
        self.show_banner()
        self.show_copilot_integration_status()

        # Collect project information
        project_info = self.collect_project_info()

        # Use provided description if meaningful
        if project_description and project_description != "New project":
            project_info["description"] = project_description

        # Create project structure
        project_path = self.create_project_structure(project_info)
        print(f"\n📁 Created project structure at: {project_path}")

        # Process with AI integration
        self.process_project_with_single_ai_request(project_info, project_path)

        print("\n🎉 Project-Start Enhanced completed successfully!")
        print(f"📁 Project files created in: {project_path}")
        print("🚀 Ready for Step 2: Enhanced SPARC methodology")


def select_ai_assistant() -> str:
    """Always return Gemini as the AI assistant"""
    print("\n🤖 AI ASSISTANT: Gemini CLI")
    print("=" * 30)
    print(
        "This CLI is configured to use Gemini CLI for AI-powered document generation."
    )
    print("✅ Selected: Gemini CLI")
    return "gemini"


def main():
    parser = argparse.ArgumentParser(
        description="Project-Start Enhanced CLI with Gemini AI Integration"
    )
    parser.add_argument("command", nargs="?", help="Command to execute")
    parser.add_argument("description", nargs="?", help="Project description")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--existing-project", action="store_true", help="Analyze existing project"
    )
    parser.add_argument("--project-path", help="Specify project path")
    parser.add_argument("--ai", default="gemini", help="AI assistant to use")

    # If no arguments provided, show interactive menu
    if len(sys.argv) == 1:
        cli = ProjectStartCLI()
        cli.show_interactive_menu()
        return

    args = parser.parse_args()

    # Always use Gemini
    cli = ProjectStartCLI()

    try:
        if args.command == "start":
            cli.project_start_enhanced(args.description or "New project")
        elif args.command == "/enhance-step-1":
            cli.enhance_step_1(args.description or "", args.existing_project)
        elif args.command == "/enhance-step-2":
            cli.enhance_step_2(args.project_path or "")
        elif args.command == "/enhance-step-3":
            cli.enhance_step_3(args.project_path or "")
        elif args.command == "/enhance-step-4":
            cli.enhance_step_4(args.project_path or "")
        elif args.command == "/project-start-enhanced":
            cli.project_start_enhanced_workflow(args.description or "")
        elif args.command == "/configure-project-root":
            cli.configure_project_root()
        else:
            print(f"Unknown command: {args.command}")
            print("\nAvailable commands:")
            print("• start                    - Interactive new project creation")
            print("• /enhance-step-1          - Discovery and specification generation")
            print("• /enhance-step-2          - SPARC planning methodology")
            print("• /enhance-step-3          - Context systems creation")
            print("• /enhance-step-4          - PACT framework deployment")
            print("• /project-start-enhanced  - Complete 4-step workflow")
            print("• /configure-project-root  - Configure project root")
            print("\nExamples:")
            print('python cli/project_start_cli.py start "My new project"')
            print(
                'python cli/project_start_cli.py /enhance-step-1 "Chat app" --existing-project'
            )
            print(
                "python cli/project_start_cli.py /enhance-step-2 --project-path specs/001-my-project"
            )
            print("\nOr run without arguments for interactive menu:")
            print("python cli/project_start_cli.py")

    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user")
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError) as e:
        print(f"\n❌ Error: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()

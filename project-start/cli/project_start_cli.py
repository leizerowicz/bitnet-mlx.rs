#!/usr/bin/env python3
"""
Project-Start Enhanced CLI - Interactive specification-driven development tool

Usage:
    python3 project_start_cli.py                     # Interactive menu
    python3 project_start_cli.py /enhance-step-1     # Step 1: Discovery
    python3 project_start_cli.py /enhance-step-2     # Step 2: SPARC Planning
    python3 project_start_cli.py /enhance-step-3     # Step 3: Context Systems
    python3 project_start_cli.py /enhance-step-4     # Step 4: PACT Framework
    python3 project_start_cli.py /project-start-enhanced  # Complete workflow
"""

import os
import sys
import argparse
import subprocess
import shutil
import json
from pathlib import Path

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


def check_gemini_cli() -> bool:
    """Check if Gemini CLI is available"""
    return shutil.which("gemini") is not None


def run_gemini_command(prompt: str, output_file: Path, context: str = "") -> bool:
    """Run Gemini CLI command to generate content"""
    try:
        cmd = ["gemini"]

        # Build the full prompt
        full_prompt = prompt
        if context:
            full_prompt = f"Context: {context}\n\n{prompt}"

        # Run gemini command
        result = subprocess.run(
            cmd, input=full_prompt, text=True, capture_output=True, check=True
        )

        # Write output to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.stdout)

        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def generate_document_with_ai(
    template_path: Path, output_path: Path, project_context: dict, document_type: str
) -> bool:
    """Generate document using AI or fallback to template"""

    # Try Gemini CLI first
    if check_gemini_cli():
        context_str = json.dumps(project_context, indent=2)
        prompt = f"""Generate a comprehensive {document_type} document based on the following project context:

{context_str}

The document should follow Project-Start constitutional framework principles:
- Specification-driven development
- Test-first methodology
- Constitutional compliance
- Agent coordination

Please provide a detailed, professional {document_type} that can serve as a foundation for development."""

        if run_gemini_command(prompt, output_path, context_str):
            print(f"  ✅ Generated {document_type} using Gemini CLI")
            return True
        else:
            print(f"  ⚠️  Gemini CLI failed, falling back to template")

    # Fallback to template
    if template_path.exists():
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template_content = f.read()

            # Simple template variable replacement
            for key, value in project_context.items():
                template_content = template_content.replace(f"{{{key}}}", str(value))

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(template_content)

            print(f"  ✅ Generated {document_type} using template")
            return True

        except Exception as e:
            print(f"  ❌ Failed to generate {document_type}: {e}")
            return False
    else:
        print(f"  ❌ Template not found: {template_path}")
        return False


class ProjectStartCLI:
    def __init__(self):
        self.project_root = self._detect_project_root()
        self.vscode_env = self._detect_vscode_environment()

    def _detect_vscode_environment(self) -> bool:
        """Detect if running in VS Code environment"""
        return (
            os.environ.get("VSCODE_PID") is not None
            or os.environ.get("TERM_PROGRAM") == "vscode"
            or "vscode" in os.environ.get("TERM_PROGRAM_VERSION", "").lower()
        )

    def _detect_project_root(self) -> Path:
        """Detect the project root directory"""
        current = Path.cwd()

        # Check for .project-start-config file
        config_file = current / ".project-start-config"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    for line in f:
                        if line.startswith("TARGET_PROJECT_ROOT="):
                            return Path(line.split("=", 1)[1].strip())
            except Exception:
                pass

        return current

    def show_banner(self):
        """Display the Project-Start banner"""
        print(BANNER)
        print(f"\n{'='*80}")
        print(f"{TAGLINE:^80}")
        print(f"{'='*80}")

        # Show AI integration status
        if check_gemini_cli():
            print(f"{'🤖 Gemini CLI Integration: ENABLED':^80}")
        else:
            print(f"{'📝 Template Mode: Gemini CLI not detected':^80}")
        print(f"{'='*80}")

    def ask_question(
        self, question: str, default: str = "", required: bool = True
    ) -> str:
        """Ask a question with optional default value"""
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
                print("This field is required. Please enter a value.")

    def ask_multiple_choice(
        self, question: str, choices: list, default: str = ""
    ) -> str:
        """Ask a multiple choice question"""
        print(f"\n{question}")
        for i, choice in enumerate(choices, 1):
            marker = " (default)" if choice == default else ""
            print(f"{i}. {choice}{marker}")

        while True:
            try:
                answer = input("\nEnter your choice (number): ").strip()
                if not answer and default:
                    return default
                choice_num = int(answer)
                if 1 <= choice_num <= len(choices):
                    return choices[choice_num - 1]
                else:
                    print(f"Please enter a number between 1 and {len(choices)}")
            except ValueError:
                print("Please enter a valid number")

    def ask_yes_no(self, question: str, default: bool = True) -> bool:
        """Ask a yes/no question"""
        default_text = "Y/n" if default else "y/N"
        answer = input(f"{question} ({default_text}): ").strip().lower()
        if not answer:
            return default
        return answer in ["y", "yes", "true", "1"]

    def show_interactive_menu(self) -> None:
        """Show interactive menu for command selection"""
        self.show_banner()

        print("\n🚀 PROJECT-START ENHANCED CLI")
        print("=" * 50)
        print("Choose an action:")
        print()
        print("1. 📋 Step 1: Discovery & Specification Generation")
        print("2. 🎯 Step 2: SPARC Planning Methodology")
        print("3. 🧠 Step 3: Context Systems Creation")
        print("4. 🤝 Step 4: PACT Framework Deployment")
        print("5. ⚡ Complete Enhanced Workflow (All Steps)")
        print("6. ⚙️  Configure Project Root")
        print("7. ❓ Help & Documentation")
        print("8. 🚪 Exit")

        while True:
            try:
                choice = input("\nEnter your choice (1-8): ").strip()

                if choice == "1":
                    self._handle_step_1()
                    break
                elif choice == "2":
                    self._handle_step_2()
                    break
                elif choice == "3":
                    self._handle_step_3()
                    break
                elif choice == "4":
                    self._handle_step_4()
                    break
                elif choice == "5":
                    self._handle_complete_workflow()
                    break
                elif choice == "6":
                    self._handle_configure_project_root()
                    break
                elif choice == "7":
                    self._show_help_documentation()
                    break
                elif choice == "8":
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("Please enter a number between 1 and 8")
            except KeyboardInterrupt:
                print("\n\n🛑 Operation cancelled by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def _handle_step_1(self) -> None:
        """Handle Step 1: Discovery & Specification Generation"""
        print("\n📋 STEP 1: DISCOVERY & SPECIFICATION GENERATION")
        print("=" * 55)

        description = self.ask_question(
            "Enter a brief project description", required=True
        )

        existing = self.ask_yes_no("Is this for an existing project?", default=False)

        self._run_step_command("/enhance-step-1", description, existing)

    def _handle_step_2(self) -> None:
        """Handle Step 2: SPARC Planning"""
        print("\n🎯 STEP 2: SPARC PLANNING METHODOLOGY")
        print("=" * 45)

        project_path = self.ask_question(
            "Enter project path (or press Enter for current directory)",
            default=str(self.project_root),
            required=False,
        )

        self._run_step_command("/enhance-step-2", project_path=project_path)

    def _handle_step_3(self) -> None:
        """Handle Step 3: Context Systems"""
        print("\n🧠 STEP 3: CONTEXT SYSTEMS CREATION")
        print("=" * 45)

        project_path = self.ask_question(
            "Enter project path (or press Enter for current directory)",
            default=str(self.project_root),
            required=False,
        )

        self._run_step_command("/enhance-step-3", project_path=project_path)

    def _handle_step_4(self) -> None:
        """Handle Step 4: PACT Framework"""
        print("\n🤝 STEP 4: PACT FRAMEWORK DEPLOYMENT")
        print("=" * 45)

        project_path = self.ask_question(
            "Enter project path (or press Enter for current directory)",
            default=str(self.project_root),
            required=False,
        )

        self._run_step_command("/enhance-step-4", project_path=project_path)

    def _handle_complete_workflow(self) -> None:
        """Handle Complete Enhanced Workflow"""
        print("\n⚡ COMPLETE ENHANCED WORKFLOW")
        print("=" * 40)

        description = self.ask_question(
            "Enter a brief project description", required=True
        )

        self._run_step_command("/project-start-enhanced", description)

    def _handle_configure_project_root(self) -> None:
        """Handle Configure Project Root"""
        print("\n⚙️  CONFIGURE PROJECT ROOT")
        print("=" * 35)

        self._run_step_command("/configure-project-root")

    def _show_help_documentation(self) -> None:
        """Show help and documentation"""
        print("\n❓ HELP & DOCUMENTATION")
        print("=" * 30)
        print()
        print("🔧 Available Commands:")
        print("  /enhance-step-1          - Discovery and specification generation")
        print("  /enhance-step-2          - SPARC planning methodology")
        print("  /enhance-step-3          - Context systems creation")
        print("  /enhance-step-4          - PACT framework deployment")
        print("  /project-start-enhanced  - Complete 4-step workflow")
        print("  /configure-project-root  - Configure project root")
        print()
        print("📚 Documentation:")
        print("  • README.md files in each step directory")
        print("  • Constitutional framework in PROJECT_START_CONSTITUTION.md")
        print("  • Memory systems in memory/ directory")
        print()
        print("🤖 AI Integration:")

        # Check Gemini CLI availability
        if check_gemini_cli():
            print("  ✅ Gemini CLI detected - Enhanced AI document generation enabled")
        else:
            print("  ⚠️  Gemini CLI not found - Using template-based generation")
            print("     Install with: pip install google-generativeai")

        print("  • Intelligent document generation with constitutional compliance")
        print("  • Fallback templates when AI tools unavailable")
        print("  • Context-aware multi-agent coordination")
        print()
        print("📋 Generated Documents:")
        print("  Step 1: BACKLOG.md, IMPLEMENTATION_GUIDE.md, RISK_ASSESSMENT.md")
        print("  Step 2: SPARC_*.md documents with methodology framework")
        print("  Step 3: copilot-instructions.md, expert_files/, agent_coordination.md")
        print("  Step 4: PACT framework documents for multi-agent testing")
        print()
        print("🏗️ Project Structure:")
        print("  specs/001-project-name/     - Generated project specifications")
        print("  ├── BACKLOG.md")
        print("  ├── IMPLEMENTATION_GUIDE.md")
        print("  ├── sparc/                  - SPARC methodology documents")
        print("  ├── .github/                - AI agent instructions")
        print("  ├── expert_files/           - Specialized expert contexts")
        print("  └── [PACT framework files]")
        print()
        input("\nPress Enter to return to main menu...")

    def _run_step_command(
        self,
        command: str,
        description: str = "",
        existing: bool = False,
        project_path: str = "",
    ):
        """Run a step command as a subprocess"""
        try:
            cmd = ["python3", "project_start_cli.py", command]

            if description:
                cmd.append(description)

            if existing:
                cmd.append("--existing-project")

            if project_path:
                cmd.extend(["--project-path", project_path])

            print(f"\n🚀 Running: {' '.join(cmd)}")
            print("=" * 50)

            # Run the command
            result = subprocess.run(cmd, cwd=self.project_root)

            if result.returncode == 0:
                print(f"\n✅ {command} completed successfully!")
            else:
                print(f"\n❌ {command} failed with return code {result.returncode}")

        except Exception as e:
            print(f"❌ Error running {command}: {e}")

    # Core step methods that call the respective enhancement scripts
    def enhance_step_1(
        self, description: str = "", existing_project: bool = False
    ) -> None:
        """Enhanced Step 1: Discovery and specification generation"""
        print("🔍 Starting Enhanced Step 1: Discovery & Specification Generation")
        print("=" * 70)

        if not description:
            description = self.ask_question(
                "Enter a brief project description", required=True
            )

        # Generate project context
        project_context = self._gather_project_context(description, existing_project)

        # Create project directory
        project_dir = self._create_project_directory(project_context["project_name"])

        # Generate specification documents
        self._generate_specification_documents(project_dir, project_context)

        print(f"\n✅ Step 1 completed successfully!")
        print(f"📁 Project specs created in: {project_dir}")

    def _gather_project_context(self, description: str, existing_project: bool) -> dict:
        """Gather comprehensive project context through questionnaire"""
        print("\n📋 GATHERING PROJECT CONTEXT")
        print("=" * 40)

        # Basic project info
        project_name = self.ask_question(
            "Project name (for directory/file naming)",
            default=description.lower().replace(" ", "-")[:30],
            required=True,
        )

        # Technical details
        tech_stack = self.ask_multiple_choice(
            "Primary technology stack",
            ["Python", "JavaScript/TypeScript", "Java", "C#", "Go", "Rust", "Other"],
            default="Python",
        )

        project_type = self.ask_multiple_choice(
            "Project type",
            [
                "Web Application",
                "CLI Tool",
                "API/Backend",
                "Desktop App",
                "Mobile App",
                "Library/Package",
                "Other",
            ],
            default="Web Application",
        )

        target_audience = self.ask_question(
            "Target audience/users", default="End users"
        )

        key_features = self.ask_question(
            "Key features (comma-separated)", required=True
        )

        # Additional context
        constraints = self.ask_question(
            "Technical constraints or requirements", required=False
        )

        success_criteria = self.ask_question(
            "Success criteria or goals", required=False
        )

        return {
            "project_name": project_name,
            "description": description,
            "tech_stack": tech_stack,
            "project_type": project_type,
            "target_audience": target_audience,
            "key_features": key_features,
            "constraints": constraints,
            "success_criteria": success_criteria,
            "existing_project": existing_project,
            "timestamp": subprocess.run(
                ["date"], capture_output=True, text=True, check=False
            ).stdout.strip(),
        }

    def _create_project_directory(self, project_name: str) -> Path:
        """Create numbered project directory in specs/"""
        specs_dir = self.project_root / "specs"
        specs_dir.mkdir(exist_ok=True)

        # Find next available project number
        existing_projects = [
            d
            for d in specs_dir.iterdir()
            if d.is_dir() and d.name.startswith(("001-", "002-", "003-"))
        ]
        next_num = len(existing_projects) + 1

        project_dir = specs_dir / f"{next_num:03d}-{project_name}"
        project_dir.mkdir(exist_ok=True)

        print(f"📁 Created project directory: {project_dir}")
        return project_dir

    def _generate_specification_documents(
        self, project_dir: Path, context: dict
    ) -> None:
        """Generate all specification documents using AI or templates"""
        print("\n📄 GENERATING SPECIFICATION DOCUMENTS")
        print("=" * 45)

        # Check AI availability
        ai_available = check_gemini_cli()
        if ai_available:
            print("🤖 Using Gemini CLI for enhanced document generation")
        else:
            print("📝 Using template-based document generation")

        documents = [
            ("BACKLOG.md", "product backlog with user stories and priorities"),
            ("IMPLEMENTATION_GUIDE.md", "technical implementation guide"),
            ("RISK_ASSESSMENT.md", "comprehensive risk assessment"),
            ("FILE_OUTLINE.md", "project file structure and organization"),
            (
                "constitutional_validation.md",
                "Project-Start constitutional compliance validation",
            ),
            (
                "clarification_needed.md",
                "stakeholder questions and clarifications needed",
            ),
        ]

        templates_dir = self.project_root / "templates"

        for filename, doc_type in documents:
            print(f"  📄 Generating {filename}...")

            output_path = project_dir / filename
            template_path = templates_dir / f"{filename}.template"

            # Use AI generation
            success = generate_document_with_ai(
                template_path, output_path, context, doc_type
            )

            if not success:
                # Create basic document if all else fails
                self._create_fallback_document(output_path, filename, context)

    def _create_fallback_document(
        self, output_path: Path, filename: str, context: dict
    ) -> None:
        """Create a basic document if AI and templates fail"""
        print(f"  📝 Creating basic {filename}...")

        basic_content = f"""# {filename.replace('.md', '').replace('_', ' ').title()}

**Project:** {context['project_name']}
**Description:** {context['description']}
**Generated:** {context['timestamp']}

## Project Context

- **Technology Stack:** {context['tech_stack']}
- **Project Type:** {context['project_type']}
- **Target Audience:** {context['target_audience']}
- **Key Features:** {context['key_features']}

## Document Content

*This document was auto-generated as a placeholder. Please enhance with specific content for {filename.replace('.md', '')}.*

## Next Steps

1. Review and enhance this document content
2. Validate against Project-Start constitutional framework
3. Coordinate with development team for implementation

---
*Generated by Project-Start Enhanced CLI*
"""

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(basic_content)
            print(f"  ✅ Created fallback {filename}")
        except Exception as e:
            print(f"  ❌ Failed to create {filename}: {e}")

    def enhance_step_2(self, project_path: str = "") -> None:
        """Enhanced Step 2: SPARC Planning"""
        print("🎯 Starting Enhanced Step 2: SPARC Planning Methodology")
        print("=" * 60)

        if not project_path:
            project_path = self.ask_question(
                "Enter project path (or press Enter for current directory)",
                default=str(self.project_root),
                required=False,
            )

        target_dir = Path(project_path) if project_path else self.project_root

        # Check if this is a specs directory with existing project
        if not self._validate_project_directory(target_dir):
            return

        # Generate SPARC documents
        self._generate_sparc_documents(target_dir)

        print("✅ Step 2 completed successfully!")

    def _validate_project_directory(self, project_dir: Path) -> bool:
        """Validate that the project directory has the required structure"""
        if not project_dir.exists():
            print(f"❌ Project directory does not exist: {project_dir}")
            return False

        # Look for project specification files
        required_files = ["BACKLOG.md", "IMPLEMENTATION_GUIDE.md"]
        missing_files = [f for f in required_files if not (project_dir / f).exists()]

        if missing_files:
            print(f"⚠️  Missing specification files: {', '.join(missing_files)}")
            print("   Run Step 1 first to generate project specifications")
            return False

        return True

    def _generate_sparc_documents(self, project_dir: Path) -> None:
        """Generate SPARC methodology documents"""
        print("\n📄 GENERATING SPARC METHODOLOGY DOCUMENTS")
        print("=" * 45)

        # Create sparc subdirectory
        sparc_dir = project_dir / "sparc"
        sparc_dir.mkdir(exist_ok=True)

        # Load existing project context
        project_context = self._load_project_context(project_dir)

        sparc_documents = [
            (
                "SPARC_SPECIFICATION.md",
                "formal specification document following SPARC methodology",
            ),
            ("SPARC_PSEUDOCODE.md", "detailed pseudocode and algorithm design"),
            ("SPARC_ARCHITECTURE.md", "system architecture and design patterns"),
            ("SPARC_REFINEMENT.md", "refinement strategies and testing approach"),
            ("SPARC_COMPLETION.md", "completion criteria and deployment procedures"),
        ]

        ai_available = check_gemini_cli()
        if ai_available:
            print("🤖 Using Gemini CLI for SPARC document generation")
        else:
            print("📝 Using template-based SPARC document generation")

        templates_dir = self.project_root / "templates"

        for filename, doc_type in sparc_documents:
            print(f"  📄 Generating {filename}...")

            output_path = sparc_dir / filename
            template_path = templates_dir / f"sparc_{filename.lower()}.template"

            # Enhanced context for SPARC documents
            sparc_context = {
                **project_context,
                "sparc_phase": filename.replace("SPARC_", "")
                .replace(".md", "")
                .lower(),
                "constitutional_framework": "Project-Start constitutional principles",
                "methodology": "SPARC (Specification, Pseudocode, Architecture, Refinement, Completion)",
            }

            success = generate_document_with_ai(
                template_path, output_path, sparc_context, f"SPARC {doc_type}"
            )

            if not success:
                self._create_sparc_fallback(output_path, filename, sparc_context)

    def _load_project_context(self, project_dir: Path) -> dict:
        """Load project context from existing specification files"""
        context = {
            "project_directory": str(project_dir),
            "timestamp": subprocess.run(
                ["date"], capture_output=True, text=True, check=False
            ).stdout.strip(),
        }

        # Try to extract context from BACKLOG.md
        backlog_file = project_dir / "BACKLOG.md"
        if backlog_file.exists():
            try:
                with open(backlog_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Simple extraction of project info
                    for line in content.split("\n"):
                        if line.startswith("**Project:**"):
                            context["project_name"] = line.split(":", 1)[1].strip()
                        elif line.startswith("**Description:**"):
                            context["description"] = line.split(":", 1)[1].strip()
            except Exception:
                pass

        return context

    def _create_sparc_fallback(
        self, output_path: Path, filename: str, context: dict
    ) -> None:
        """Create fallback SPARC document"""
        phase = filename.replace("SPARC_", "").replace(".md", "")

        content = f"""# {phase} Phase - SPARC Methodology

**Project:** {context.get('project_name', 'Unknown')}
**Phase:** {phase}
**Generated:** {context['timestamp']}

## Constitutional Framework Compliance

This document follows Project-Start constitutional principles:
- Specification-driven development
- Test-first methodology
- Agent coordination protocols
- Quality assurance gates

## {phase} Overview

*This section should contain the detailed {phase.lower()} content according to SPARC methodology.*

## Implementation Notes

- Follow constitutional framework requirements
- Maintain specification-driven approach
- Coordinate with multi-agent systems
- Validate against quality gates

## Next Steps

1. Review and enhance this {phase.lower()} content
2. Validate constitutional compliance
3. Coordinate with development team
4. Proceed to next SPARC phase

---
*Generated by Project-Start Enhanced CLI with SPARC Methodology*
"""

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✅ Created fallback {filename}")
        except Exception as e:
            print(f"  ❌ Failed to create {filename}: {e}")

    def enhance_step_3(self, project_path: str = "") -> None:
        """Enhanced Step 3: Context Systems"""
        print("🧠 Starting Enhanced Step 3: Context Systems Creation")
        print("=" * 55)

        if not project_path:
            project_path = self.ask_question(
                "Enter project path (or press Enter for current directory)",
                default=str(self.project_root),
                required=False,
            )

        target_dir = Path(project_path) if project_path else self.project_root

        if not self._validate_project_directory(target_dir):
            return

        # Generate context systems
        self._generate_context_systems(target_dir)

        print("✅ Step 3 completed successfully!")

    def _generate_context_systems(self, project_dir: Path) -> None:
        """Generate AI agent context systems"""
        print("\n🧠 GENERATING CONTEXT SYSTEMS")
        print("=" * 35)

        # Load project context
        project_context = self._load_project_context(project_dir)

        # Create .github directory for copilot instructions
        github_dir = project_dir / ".github"
        github_dir.mkdir(exist_ok=True)

        # Create expert_files directory
        expert_dir = project_dir / "expert_files"
        expert_dir.mkdir(exist_ok=True)

        ai_available = check_gemini_cli()
        if ai_available:
            print("🤖 Using Gemini CLI for context system generation")
        else:
            print("📝 Using template-based context system generation")

        # Generate copilot instructions
        self._generate_copilot_instructions(github_dir, project_context)

        # Generate expert files
        self._generate_expert_files(expert_dir, project_context)

        # Generate agent coordination
        self._generate_agent_coordination(project_dir, project_context)

    def _generate_copilot_instructions(self, github_dir: Path, context: dict) -> None:
        """Generate GitHub Copilot instructions"""
        print("  📄 Generating copilot-instructions.md...")

        output_path = github_dir / "copilot-instructions.md"
        template_path = (
            self.project_root / "templates" / "constitutional_copilot_instructions.md"
        )

        copilot_context = {
            **context,
            "agent_type": "GitHub Copilot",
            "framework": "Project-Start Constitutional Framework",
            "role": "Primary development assistant with constitutional compliance",
        }

        success = generate_document_with_ai(
            template_path,
            output_path,
            copilot_context,
            "GitHub Copilot instructions with constitutional framework integration",
        )

        if not success:
            self._create_copilot_fallback(output_path, copilot_context)

    def _generate_expert_files(self, expert_dir: Path, context: dict) -> None:
        """Generate specialized expert context files"""
        experts = [
            (
                "architecture_expert.md",
                "software architecture and system design expert",
            ),
            ("tech_stack_expert.md", "technology stack and implementation expert"),
            (
                "methodology_expert.md",
                "Project-Start methodology and constitutional framework expert",
            ),
        ]

        templates_dir = self.project_root / "templates"

        for filename, expert_type in experts:
            print(f"  📄 Generating {filename}...")

            output_path = expert_dir / filename
            template_path = templates_dir / f"expert_{filename}"

            expert_context = {
                **context,
                "expert_specialization": expert_type,
                "constitutional_role": f"Constitutional {expert_type} with Project-Start compliance",
            }

            success = generate_document_with_ai(
                template_path,
                output_path,
                expert_context,
                f"Expert context file for {expert_type}",
            )

            if not success:
                self._create_expert_fallback(output_path, filename, expert_context)

    def _generate_agent_coordination(self, project_dir: Path, context: dict) -> None:
        """Generate agent coordination protocols"""
        print("  📄 Generating agent_coordination.md...")

        output_path = project_dir / "agent_coordination.md"
        template_path = self.project_root / "templates" / "agent_coordination.md"

        coordination_context = {
            **context,
            "coordination_type": "Multi-agent coordination protocols",
            "framework_compliance": "Project-Start constitutional framework",
        }

        success = generate_document_with_ai(
            template_path,
            output_path,
            coordination_context,
            "multi-agent coordination protocols and workflows",
        )

        if not success:
            self._create_coordination_fallback(output_path, coordination_context)

    def _create_copilot_fallback(self, output_path: Path, context: dict) -> None:
        """Create fallback copilot instructions"""
        content = f"""# Copilot Instructions

**Project:** {context.get('project_name', 'Unknown')}
**Framework:** Project-Start Constitutional Framework
**Generated:** {context['timestamp']}

## Constitutional Principles

Follow these immutable principles:

1. **Specification-Driven Development** - Always work from specifications
2. **Test-First Methodology** - Write tests before implementation
3. **Constitutional Compliance** - Validate against framework rules
4. **Agent Coordination** - Coordinate with other AI agents

## Project Context

- **Description:** {context.get('description', 'See project documentation')}
- **Technology Stack:** {context.get('tech_stack', 'See implementation guide')}

## Instructions

When working on this project:

1. Always review existing specifications before coding
2. Follow constitutional framework principles
3. Coordinate with other expert agents
4. Maintain quality gates and validation

## Code Style

- Follow project conventions
- Write comprehensive tests
- Document all decisions
- Validate constitutional compliance

---
*Generated by Project-Start Enhanced CLI*
"""

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print("  ✅ Created fallback copilot-instructions.md")
        except Exception as e:
            print(f"  ❌ Failed to create copilot instructions: {e}")

    def _create_expert_fallback(
        self, output_path: Path, filename: str, context: dict
    ) -> None:
        """Create fallback expert file"""
        expert_name = filename.replace("_expert.md", "").replace("_", " ").title()

        content = f"""# {expert_name} Expert Context

**Project:** {context.get('project_name', 'Unknown')}
**Specialization:** {context.get('expert_specialization', expert_name)}
**Generated:** {context['timestamp']}

## Expert Role

As a {expert_name} expert, provide guidance on:

- {expert_name} best practices
- Constitutional framework compliance
- Integration with Project-Start methodology
- Coordination with other expert agents

## Constitutional Framework

Ensure all recommendations follow:
- Specification-driven development
- Test-first methodology
- Quality assurance gates
- Agent coordination protocols

## Expertise Areas

*Add specific {expert_name.lower()} expertise areas and guidelines*

---
*Generated by Project-Start Enhanced CLI*
"""

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✅ Created fallback {filename}")
        except Exception as e:
            print(f"  ❌ Failed to create {filename}: {e}")

    def _create_coordination_fallback(self, output_path: Path, context: dict) -> None:
        """Create fallback agent coordination document"""
        content = f"""# Agent Coordination Protocols

**Project:** {context.get('project_name', 'Unknown')}
**Framework:** Project-Start Constitutional Framework
**Generated:** {context['timestamp']}

## Multi-Agent Coordination

This document defines coordination protocols between AI agents working on this project.

## Agent Roles

1. **Primary Agent (Copilot)** - Main development assistant
2. **Architecture Expert** - System design and architecture
3. **Technology Expert** - Implementation and tech stack
4. **Methodology Expert** - Framework compliance and process

## Coordination Protocols

### Communication Standards
- Use clear, specification-driven language
- Reference constitutional framework principles
- Maintain context across agent interactions

### Decision Making
- Defer to specifications and constitutional framework
- Coordinate on architectural decisions
- Validate against quality gates

### Workflow Integration
- Follow Project-Start 4-step methodology
- Maintain SPARC process compliance
- Coordinate testing and validation

## Constitutional Compliance

All agents must ensure:
- Specification-driven development
- Test-first methodology
- Quality assurance
- Framework compliance validation

---
*Generated by Project-Start Enhanced CLI*
"""

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print("  ✅ Created fallback agent_coordination.md")
        except Exception as e:
            print(f"  ❌ Failed to create agent coordination: {e}")

    def enhance_step_4(self, project_path: str = "") -> None:
        """Enhanced Step 4: PACT Framework"""
        print("🤝 Starting Enhanced Step 4: PACT Framework Deployment")
        print("=" * 60)

        if not project_path:
            project_path = self.ask_question(
                "Enter project path (or press Enter for current directory)",
                default=str(self.project_root),
                required=False,
            )

        target_dir = Path(project_path) if project_path else self.project_root

        if not self._validate_project_directory(target_dir):
            return

        # Generate PACT framework
        self._generate_pact_framework(target_dir)

        print("✅ Step 4 completed successfully!")

    def _generate_pact_framework(self, project_dir: Path) -> None:
        """Generate PACT framework documents"""
        print("\n🤝 GENERATING PACT FRAMEWORK")
        print("=" * 35)

        # Load project context
        project_context = self._load_project_context(project_dir)

        ai_available = check_gemini_cli()
        if ai_available:
            print("🤖 Using Gemini CLI for PACT framework generation")
        else:
            print("📝 Using template-based PACT framework generation")

        pact_documents = [
            (
                "AGENT_ECOSYSTEM_DESIGN.md",
                "multi-agent ecosystem design and architecture",
            ),
            ("COORDINATION_STRATEGY.md", "agent coordination strategy and protocols"),
            ("COLLABORATIVE_WORKFLOWS.md", "collaborative workflow definitions"),
            ("AGENTIC_TESTING_FRAMEWORK.md", "comprehensive agentic testing framework"),
        ]

        templates_dir = self.project_root / "templates"

        for filename, doc_type in pact_documents:
            print(f"  📄 Generating {filename}...")

            output_path = project_dir / filename
            template_path = templates_dir / f"pact_{filename.lower()}.template"

            pact_context = {
                **project_context,
                "framework_type": "PACT (Planning, Action, Coordination, Testing)",
                "constitutional_compliance": "Project-Start constitutional framework",
                "document_purpose": doc_type,
            }

            success = generate_document_with_ai(
                template_path, output_path, pact_context, f"PACT framework {doc_type}"
            )

            if not success:
                self._create_pact_fallback(output_path, filename, pact_context)

    def _create_pact_fallback(
        self, output_path: Path, filename: str, context: dict
    ) -> None:
        """Create fallback PACT framework document"""
        doc_type = (
            filename.replace("AGENTIC_", "")
            .replace("_", " ")
            .replace(".md", "")
            .title()
        )

        content = f"""# {doc_type} - PACT Framework

**Project:** {context.get('project_name', 'Unknown')}
**Framework:** PACT (Planning, Action, Coordination, Testing)
**Document Type:** {context.get('document_purpose', doc_type)}
**Generated:** {context['timestamp']}

## Constitutional Framework Compliance

This document implements Project-Start constitutional principles:
- Specification-driven development
- Test-first methodology
- Agent coordination protocols
- Quality assurance gates

## PACT Framework Overview

The PACT framework provides:
- **Planning**: Strategic planning and specification
- **Action**: Implementation and execution
- **Coordination**: Multi-agent coordination
- **Testing**: Comprehensive testing strategies

## {doc_type} Implementation

*This section should contain detailed {doc_type.lower()} implementation according to PACT framework.*

## Multi-Agent Coordination

- Define agent roles and responsibilities
- Establish communication protocols
- Implement coordination mechanisms
- Ensure constitutional compliance

## Quality Assurance

- Validate against constitutional framework
- Implement testing strategies
- Monitor compliance and quality
- Coordinate validation across agents

## Next Steps

1. Review and enhance {doc_type.lower()} content
2. Validate constitutional compliance
3. Coordinate with multi-agent ecosystem
4. Implement testing and validation

---
*Generated by Project-Start Enhanced CLI with PACT Framework*
"""

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✅ Created fallback {filename}")
        except Exception as e:
            print(f"  ❌ Failed to create {filename}: {e}")

    def project_start_enhanced_workflow(self, description: str = "") -> None:
        """Complete enhanced workflow"""
        print("⚡ Starting Complete Enhanced Workflow")
        print("=" * 45)

        if not description:
            description = self.ask_question(
                "Enter a brief project description", required=True
            )

        try:
            print("\n🔍 STEP 1: Discovery & Specification Generation")
            print("=" * 55)
            self.enhance_step_1(description, False)

            # Find the most recently created project directory
            specs_dir = self.project_root / "specs"
            if specs_dir.exists():
                project_dirs = [d for d in specs_dir.iterdir() if d.is_dir()]
                if project_dirs:
                    # Get the most recent project directory
                    latest_project = max(project_dirs, key=lambda x: x.stat().st_mtime)
                    project_path = str(latest_project)

                    print(f"\n🎯 STEP 2: SPARC Planning Methodology")
                    print("=" * 45)
                    self.enhance_step_2(project_path)

                    print(f"\n🧠 STEP 3: Context Systems Creation")
                    print("=" * 40)
                    self.enhance_step_3(project_path)

                    print(f"\n🤝 STEP 4: PACT Framework Deployment")
                    print("=" * 40)
                    self.enhance_step_4(project_path)

                    print("\n🎉 COMPLETE WORKFLOW FINISHED!")
                    print("=" * 40)
                    print(f"📁 Project created in: {latest_project}")
                    print("✅ All 4 steps completed successfully!")
                    print("\n📋 Next Steps:")
                    print("1. Review generated specifications and documents")
                    print("2. Validate constitutional framework compliance")
                    print("3. Begin implementation following SPARC methodology")
                    print("4. Coordinate with AI agents using generated context")
                else:
                    print("❌ No project directory found after Step 1")
            else:
                print("❌ Specs directory not found")

        except Exception as e:
            print(f"❌ Complete workflow failed: {e}")
            print("💡 Try running individual steps to isolate the issue")

    def configure_project_root(self) -> None:
        """Configure project root"""
        print("⚙️ Configuring Project Root")
        print("=" * 30)

        try:
            subprocess.run(
                ["./scripts/configure-project-root.sh"],
                cwd=self.project_root,
                check=True,
            )
            print("✅ Project root configured successfully!")

        except subprocess.CalledProcessError as e:
            print(f"❌ Configuration failed: {e}")
        except FileNotFoundError:
            print("❌ configure-project-root.sh script not found.")


def main():
    parser = argparse.ArgumentParser(
        description="Project-Start Enhanced CLI - Specification-Driven Development"
    )
    parser.add_argument("command", nargs="?", help="Command to execute")
    parser.add_argument("description", nargs="?", help="Project description")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--existing-project", action="store_true", help="Analyze existing project"
    )
    parser.add_argument("--project-path", help="Specify project path")

    # If no arguments provided, show interactive menu
    if len(sys.argv) == 1:
        cli = ProjectStartCLI()
        cli.show_interactive_menu()
        return

    args = parser.parse_args()
    cli = ProjectStartCLI()

    try:
        if args.command == "/enhance-step-1":
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
            print(f"❌ Unknown command: {args.command}")
            print("\n🔧 Available commands:")
            print("• /enhance-step-1          - Discovery and specification generation")
            print("• /enhance-step-2          - SPARC planning methodology")
            print("• /enhance-step-3          - Context systems creation")
            print("• /enhance-step-4          - PACT framework deployment")
            print("• /project-start-enhanced  - Complete 4-step workflow")
            print("• /configure-project-root  - Configure project root")
            print("\n💡 Examples:")
            print('python3 project_start_cli.py /enhance-step-1 "Chat app"')
            print(
                "python3 project_start_cli.py /enhance-step-2 --project-path specs/001-my-project"
            )
            print("\n🚀 Or run without arguments for interactive menu:")
            print("python3 project_start_cli.py")

    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()

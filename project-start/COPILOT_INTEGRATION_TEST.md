# AI Integration Test Examples

This document demonstrates testing the updated AI integration following the GitHub Spec-Kit pattern.

## Test Scenarios

### 1. GitHub Copilot Integration Test

```bash
cd /home/runner/work/bitnet-rust/bitnet-rust/project-start/cli
python3 project_start_cli.py start "AI-powered task manager" --ai copilot
```

**Expected Output:**
- Display Agentic Engineering banner
- Show "🤖 AI Integration: ✅ GitHub Copilot ENABLED"
- Detect environment and tool availability
- Guide through interactive questionnaire
- Generate 6 documents with GitHub Copilot integration
- Create fallback templates if VS Code not detected

### 2. Claude Code Integration Test

```bash
python3 project_start_cli.py start "Real-time chat app" --ai claude
```

**Expected Output:**
- Show "🤖 AI Integration: ✅ Claude Code ENABLED"
- Check for Claude CLI availability
- Either use Claude CLI or provide fallback templates
- Generate project documents with Claude-specific prompts

### 3. Gemini CLI Integration Test

```bash
python3 project_start_cli.py start "E-commerce platform" --ai gemini
```

**Expected Output:**
- Show "🤖 AI Integration: ✅ Gemini CLI ENABLED"
- Check for Gemini CLI availability
- Either use Gemini CLI or provide fallback templates
- Generate project documents with Gemini-specific prompts

### 4. Interactive AI Selection Test

```bash
python3 project_start_cli.py start "Blog platform"
```

**Expected Output:**
- Display AI assistant selection menu
- Show options: 1. GitHub Copilot, 2. Claude Code, 3. Gemini CLI
- Default to GitHub Copilot if user presses Enter
- Proceed with selected AI assistant

## Generated Documents

Each test should generate comprehensive project documentation:

### Core Documents
- **BACKLOG.md**: AI-generated user stories and feature priorities
- **IMPLEMENTATION_GUIDE.md**: Technology-specific development guidance  
- **RISK_ASSESSMENT.md**: Project-specific risk analysis and mitigation
- **FILE_OUTLINE.md**: Intelligent file structure based on tech stack
- **constitutional_validation.md**: AI-assisted compliance checking
- **clarification_needed.md**: Intelligent gap analysis and questions

### Document Headers

Each generated document includes:
```markdown
# Document Title

🤖 **Generated with [AI Assistant] Integration** (Project-Start CLI)

## AI Generation Context
- **AI Assistant**: [Selected Assistant]
- **Document Type**: [document_type]
- **Environment**: [VS Code Detected/Command Line]
- **Generated**: [timestamp]
```

## Key Features Demonstrated

### 1. Multi-AI Support
- Choice between GitHub Copilot, Claude Code, Gemini CLI
- Consistent interface regardless of AI assistant
- Proper tool detection and availability checking

### 2. GitHub Spec-Kit Pattern Compliance
- Real API calls instead of simulation
- Proper error handling and fallbacks
- Interactive selection with sensible defaults
- Environment-aware behavior

### 3. Enhanced User Experience
- Clear status reporting and feedback
- Smart environment detection (VS Code, AI tools)
- Graceful degradation when AI tools unavailable
- Comprehensive help and usage information

### 4. Production-Ready Implementation
- Timeout handling for AI API calls
- Comprehensive error recovery
- Cross-platform compatibility
- Secure subprocess handling

## Environment Setup for Testing

### GitHub Copilot
```bash
# Use VS Code with Copilot extension
export VSCODE_PID=12345  # Set automatically by VS Code
# OR
export GITHUB_COPILOT_ENABLED=true
```

### Claude Code
```bash
# Install Claude CLI
curl -fsSL https://docs.anthropic.com/install.sh | sh
claude --version
```

### Gemini CLI
```bash
# Install Gemini CLI
npm install -g @google-ai/gemini-cli
gemini --version
```

## Validation Checklist

- [ ] CLI shows proper AI integration status
- [ ] Tool availability is correctly detected
- [ ] AI assistant selection works (interactive and --ai flag)
- [ ] Documents are generated with AI-specific headers
- [ ] Fallback generation works when AI tools unavailable
- [ ] Error handling is graceful and informative
- [ ] Help output shows AI options correctly
- [ ] All existing functionality is preserved

## Performance Expectations

- **Startup Time**: <2 seconds for CLI initialization
- **AI Tool Detection**: <1 second for availability checking
- **Document Generation**: 30-60 seconds per document (depending on AI response time)
- **Fallback Generation**: <5 seconds for template creation
- **Memory Usage**: <100MB during operation

---

*Testing framework aligned with GitHub Spec-Kit pattern for AI integration*

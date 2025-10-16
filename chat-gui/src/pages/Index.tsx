import { useState } from "react";
import ChatMessage from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import ModelInfo from "@/components/ModelInfo";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Github } from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const Index = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hello! I'm powered by BitNet-MLX, a high-performance neural network with 1.58-bit quantization optimized for Apple Silicon. How can I help you today?",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = (content: string) => {
    const userMessage: Message = { role: "user", content };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    // Simulate AI response
    setTimeout(() => {
      const aiMessage: Message = {
        role: "assistant",
        content: "This is a demo response. In a real implementation, this would connect to the BitNet-MLX backend via API calls to the Rust inference engine.",
      };
      setMessages((prev) => [...prev, aiMessage]);
      setIsLoading(false);
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-gradient-bg flex flex-col">
      {/* Header */}
      <header className="border-b border-border bg-background/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-primary flex items-center justify-center shadow-glow">
                <span className="text-sm font-bold text-primary-foreground">B</span>
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-primary bg-clip-text text-transparent">
                  BitNet-MLX Chat
                </h1>
                <p className="text-xs text-muted-foreground">
                  1.58-bit Neural Network
                </p>
              </div>
            </div>
            <a
              href="https://github.com/leizerowicz/bitnet-mlx.rs"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-primary transition-colors"
            >
              <Github className="w-5 h-5" />
            </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 container mx-auto px-4 py-6 flex gap-6">
        {/* Chat Area */}
        <div className="flex-1 flex flex-col gap-4">
          <ScrollArea className="flex-1 rounded-2xl bg-background/30 backdrop-blur-sm border border-border">
            <div className="py-4">
              {messages.map((message, index) => (
                <ChatMessage
                  key={index}
                  role={message.role}
                  content={message.content}
                />
              ))}
              {isLoading && (
                <div className="flex gap-4 p-6 animate-fade-in">
                  <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-gradient-primary flex items-center justify-center shadow-glow animate-pulse-glow">
                    <span className="text-sm text-primary-foreground">B</span>
                  </div>
                  <div className="bg-card border border-border rounded-2xl p-4">
                    <div className="flex gap-2">
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:0.2s]" />
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:0.4s]" />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

          <ChatInput onSend={handleSendMessage} disabled={isLoading} />
        </div>

        {/* Sidebar */}
        <aside className="w-80 hidden lg:block">
          <ModelInfo />
        </aside>
      </div>
    </div>
  );
};

export default Index;

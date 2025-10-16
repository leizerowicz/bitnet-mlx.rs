import { Cpu, Zap, Database } from "lucide-react";
import { Card } from "@/components/ui/card";

const ModelInfo = () => {
  return (
    <Card className="p-6 bg-card border-border">
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-primary animate-pulse-glow" />
          <h2 className="text-lg font-semibold bg-gradient-primary bg-clip-text text-transparent">
            BitNet-MLX Model
          </h2>
        </div>
        
        <div className="space-y-3">
          <div className="flex items-center gap-3 text-sm">
            <Database className="w-4 h-4 text-primary" />
            <div>
              <div className="text-muted-foreground">Quantization</div>
              <div className="font-medium">1.58-bit</div>
            </div>
          </div>
          
          <div className="flex items-center gap-3 text-sm">
            <Cpu className="w-4 h-4 text-primary" />
            <div>
              <div className="text-muted-foreground">Backend</div>
              <div className="font-medium">Metal + MLX</div>
            </div>
          </div>
          
          <div className="flex items-center gap-3 text-sm">
            <Zap className="w-4 h-4 text-primary" />
            <div>
              <div className="text-muted-foreground">Optimization</div>
              <div className="font-medium">Apple Silicon</div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default ModelInfo;

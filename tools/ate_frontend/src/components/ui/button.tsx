import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center rounded-[6px] border text-[12px] font-medium transition-all duration-200 disabled:opacity-50",
  {
    variants: {
      variant: {
        default:
          "border-[var(--line-bright)] bg-[linear-gradient(180deg,rgba(255,255,255,0.04),transparent),var(--panel)] text-[var(--text)] hover:border-[rgba(107,193,242,0.45)] hover:text-[var(--cyan)]",
        ghost: "border-transparent text-[var(--muted)] hover:text-[var(--text)] hover:bg-white/[0.03]",
      },
      size: {
        default: "h-8 px-3.5",
        sm: "h-7 px-2.5",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  },
);
Button.displayName = "Button";

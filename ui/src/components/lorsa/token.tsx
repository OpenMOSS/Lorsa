import { cn } from "@/lib/utils";
import { Token } from "@/types/head";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "../ui/hover-card";
import { Separator } from "../ui/separator";
import { Fragment } from "react/jsx-runtime";
import { getAccentClassname } from "@/utils/style";

export type TokenInfoProps = {
  token: Token;
  maxHeadAct: number;
  position: number;
};

export const TokenInfo = ({ token, maxHeadAct, position }: TokenInfoProps) => {
  return (
    <div className="grid grid-cols-2 gap-2">
      <div className="text-sm font-bold">Token:</div>
      <div className="text-sm underline whitespace-pre-wrap">{token.token}</div>
      <div className="text-sm font-bold">Position:</div>
      <div className="text-sm">{position}</div>
      <div className="text-sm font-bold">Activation:</div>
      <div className="text-sm">{token.headAct.toFixed(3)}</div>
    </div>
  );
};

export type SuperTokenProps = {
  tokens: Token[];
  position: number;
  maxHeadAct: number;
  sampleMaxHeadAct: number;
  selectedDfa: number[] | null;
  onHoverStart: () => void;
  onHoverEnd: () => void;
};

const getDfaColor = (value: number) => {
  if (value === 0) return "";
  const opacity = Math.min(Math.floor(value * 1000), 90);
  const validOpacity = Math.max(0, Math.floor(opacity / 10) * 10);
  return `bg-green-500/${validOpacity}`;
};

export const SuperToken = ({
  tokens,
  position,
  maxHeadAct,
  sampleMaxHeadAct,
  selectedDfa,
  onHoverStart,
  onHoverEnd
}: SuperTokenProps) => {
  const displayText = tokens.map(token => token.token).join("");
  const superTokenMaxHeadAct = Math.max(...tokens.map((t) => t.headAct));

  const SuperTokenInner = () => {
    const baseIndex = tokens[0].originalIndex;
    
    return (
      <span
        className={cn(
          "underline decoration-slate-400 decoration-1 decoration-dotted underline-offset-[6px]",
          superTokenMaxHeadAct > 0 && "hover:shadow-lg hover:text-gray-600 cursor-pointer",
          sampleMaxHeadAct > 0 && superTokenMaxHeadAct === sampleMaxHeadAct && "font-bold",
          
          !selectedDfa && getAccentClassname(superTokenMaxHeadAct, maxHeadAct, "bg"),
          ...tokens.map((token, i) => {
            const globalIndex = baseIndex + i;
            return selectedDfa?.[globalIndex] !== undefined 
              ? getDfaColor(selectedDfa[globalIndex])
              : "";
          }),
          tokens.some((t) => t.isQPosition) && "bg-green-500"
        )}
        onMouseEnter={onHoverStart}
        onMouseLeave={onHoverEnd}
      >
        {displayText}
      </span>
    );
  };

  if (superTokenMaxHeadAct === 0) {
    return <SuperTokenInner />;
  }

  return (
    <HoverCard>
      <HoverCardTrigger onMouseEnter={onHoverStart} onMouseLeave={onHoverEnd}>
        <SuperTokenInner />
      </HoverCardTrigger>
      <HoverCardContent className="w-[300px] text-wrap flex flex-col gap-4">
        {tokens.length > 1 && (
          <div className="text-sm font-bold">
            This super token is composed of the {tokens.length} tokens below:
          </div>
        )}
        {tokens.map((token, i) => (
          <Fragment key={i}>
            <TokenInfo token={token} maxHeadAct={maxHeadAct} position={position + i} />
            {i < tokens.length - 1 && <Separator />}
          </Fragment>
        ))}
      </HoverCardContent>
    </HoverCard>
  );
};
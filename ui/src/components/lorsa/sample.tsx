import { Head, Sample, Token } from "@/types/head";
import { SuperToken } from "./token";
import { useState } from "react";
import { AppPagination } from "../ui/pagination";
import { Accordion, AccordionTrigger, AccordionContent, AccordionItem } from "../ui/accordion";

export const HeadSampleGroup = ({
  head,
  samples,
}: {
  head: Head;
  samples: Head["samples"];
}) => {
  const [page, setPage] = useState<number>(1);
  const maxPage = Math.ceil(samples.length / 10);

  return (
    <div className="flex flex-col gap-4 mt-4">
      <p className="font-bold">Max Activation: {Math.max(...samples[0].headActs).toFixed(3)}</p>
      {samples.slice((page - 1) * 10, page * 10).map((sample, i) => (
        <HeadActivationSample
          key={i}
          sample={sample}
          sampleName={`Sample ${(page - 1) * 10 + i + 1}`}
          maxHeadAct={head.maxHeadAct}
        />
      ))}
      <AppPagination page={page} setPage={setPage} maxPage={maxPage} />
    </div>
  );
};

export type HeadActivationSampleProps = {
  sample: Sample;
  sampleName: string;
  maxHeadAct: number;
};

export const HeadActivationSample = ({ sample, sampleName, maxHeadAct }: HeadActivationSampleProps) => {
  const [selectedDfa, setSelectedDfa] = useState<number[] | null>(null);
  const sampleMaxHeadAct = Math.max(...sample.headActs);

  const start = Math.max(0);
  const end = Math.min(sample.context.length);

  // 添加 originalIndex 记录原始位置
  const tokens = sample.context.slice(start, end).map((token, i) => ({
    token,
    headAct: sample.headActs[start + i],
    dfa: sample.dfa[start + i],
    isQPosition: false,
    originalIndex: start + i,
  }));

  console.log(tokens)

  const [tokenGroups, _] = tokens.reduce<[Token[][], Token[]]>(
    ([groups, currentGroup], token) => {
      const newGroup = [...currentGroup, token];
      try {
        const decoded = decodeURIComponent(escape(newGroup.join('')));
        return [[...groups, newGroup], []];
      } catch {
        return [groups, newGroup];
      }
    },
    [[], []]
  );

  const tokenGroupPositions = tokenGroups.reduce<number[]>(
    (acc, tokenGroup) => {
      const tokenCount = tokenGroup.length;
      return [...acc, acc[acc.length - 1] + tokenCount];
    },
    [0]
  );

  const tokensList = tokens.map((t) => t.headAct);
  const startTrigger = Math.max(tokensList.indexOf(Math.max(...tokensList)) - 100, 0);
  const endTrigger = Math.min(tokensList.indexOf(Math.max(...tokensList)) + 10, sample.context.length);
  const tokensTrigger = sample.context.slice(startTrigger, endTrigger).map((token, i) => ({
    token,
    headAct: sample.headActs[startTrigger + i],
    isQPosition: sample.qPosition === startTrigger + i,
  }));

  const [tokenGroupsTrigger, __] = tokensTrigger.reduce<[Token[][], Token[]]>(
    ([groups, currentGroup], token) => {
      const newGroup = [...currentGroup, token];
      try {
        const decoded = decodeURIComponent(escape(newGroup.join('')));
        return [[...groups, newGroup], []]; // Successfully decoded, finalize the group
      } catch {
        return [groups, newGroup];
      }
    },
    [[], []]
  );

  // console.log(tokenGroupsTrigger)

  const tokenGroupPositionsTrigger = tokenGroupsTrigger.reduce<number[]>(
    (acc, tokenGroup) => {
      const tokenCount = tokenGroup.length;
      return [...acc, acc[acc.length - 1] + tokenCount];
    },
    [0]
  );

  const handleHoverStart = (dfaArray: number[]) => setSelectedDfa(dfaArray);
  const handleHoverEnd = () => setSelectedDfa(null);

  return (
    <div>
      <Accordion type="single" collapsible>
        <AccordionItem value={sampleMaxHeadAct.toString()}>
          <AccordionTrigger>
            <div className="block text-left">
              {sampleName && <span className="text-gray-700 font-bold whitespace-pre">{sampleName}: </span>}
              {startTrigger != 0 && <span className="text-sky-300">...</span>}
              {tokenGroupsTrigger.map((tokens, i) => (
                <SuperToken
                  key={`trigger-group-${i}`}
                  tokens={tokens}
                  position={tokenGroupPositionsTrigger[i]}
                  maxHeadAct={maxHeadAct}
                  sampleMaxHeadAct={sampleMaxHeadAct}
                  selectedDfa={selectedDfa}
                  onHoverStart={() => handleHoverStart(tokens[0].dfa)}
                  onHoverEnd={handleHoverEnd}
                />
              ))}
              {endTrigger != 0 && <span className="text-sky-300"> ...</span>}
            </div>
          </AccordionTrigger>
          <AccordionContent>
            {tokenGroups.map((tokens, i) => (
              <SuperToken
                key={`group-${i}`}
                tokens={tokens}
                position={tokenGroupPositions[i]}
                maxHeadAct={maxHeadAct}
                sampleMaxHeadAct={sampleMaxHeadAct}
                selectedDfa={selectedDfa}
                onHoverStart={() => handleHoverStart(tokens[0].dfa)}
                onHoverEnd={handleHoverEnd}
              />
            ))}
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
};
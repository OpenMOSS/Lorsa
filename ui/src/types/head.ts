import { z } from "zod";

export const SampleSchema = z.object({
  context: z.array(z.string()),
  headActs: z.array(z.number()),
  dfa: z.array(z.array(z.number()))
  // qPosition: z.number(),
});

export type Sample = z.infer<typeof SampleSchema>;

export const InterpretationSchema = z.object({
  text: z.string(),
});

export type Interpretation = z.infer<typeof InterpretationSchema>;

export const HeadSchema = z.object({
  headIndex: z.number(),
  lorsaName: z.string(),
  actTimes: z.number(),
  maxHeadAct: z.number(),
  samples: z.array(SampleSchema),
  interpretation: InterpretationSchema.nullable(),
});

export type Head = z.infer<typeof HeadSchema>;

export type Token = {
  token: string;
  headAct: number;
  dfa: number[];
  isQPosition?: boolean;
  originalIndex: number; // 新增字段
};
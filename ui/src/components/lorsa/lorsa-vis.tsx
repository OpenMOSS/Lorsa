import { Head, SampleSchema } from "@/types/head";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { useState } from "react";
import Plot from "react-plotly.js";
import { useAsyncFn } from "react-use";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { Textarea } from "../ui/textarea";
import { HeadActivationSample, HeadSampleGroup } from "./sample";

const HeadCustomInputArea = ({ head }: { head: Head }) => {
  const [customInput, setCustomInput] = useState<string>("");
  const [state, submit] = useAsyncFn(async () => {
    if (!customInput) {
      alert("Please enter your input.");
      return;
    }
    return await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/lorsas/${head.lorsaName}/heads/${
        head.headIndex
      }/custom?input_text=${encodeURIComponent(customInput)}`,
      {
        method: "POST",
        headers: {
          Accept: "application/x-msgpack",
        },
      }
    )
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.arrayBuffer())
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .then((res) => decode(new Uint8Array(res)) as any)
      .then((res) =>
        camelcaseKeys(res, {
          deep: true,
          stopPaths: ["context"],
        })
      )
      .then((res) => SampleSchema.parse(res));
  }, [customInput]);

  return (
    <div className="flex flex-col gap-4">
      <p className="font-bold">Custom Input</p>
      <Textarea
        placeholder="Type your custom input here."
        value={customInput}
        onChange={(e) => setCustomInput(e.target.value)}
      />
      <Button onClick={submit} disabled={state.loading}>
        Submit
      </Button>
      {state.error && <p className="text-red-500">{state.error.message}</p>}
      {state.value && (
        <>
          <HeadActivationSample
            sample={state.value}
            sampleName="Custom Input"
            maxHeadAct={head.maxHeadAct}
          />
          <p className="font-bold">Custom Input Max Activation: {Math.max(...state.value.headActs).toFixed(3)}</p>
        </>
      )}
    </div>
  );
};

export const LorsaHeadCard = ({ head }: { head: Head }) => {

  const [showCustomInput, setShowCustomInput] = useState<boolean>(false);

  return (
    <Card id="Interp." className="container">
      <CardHeader>
        <CardTitle className="flex justify-between items-center text-xl">
          <span>
            #{head.headIndex}{" "}
            <span className="font-medium">
              (Activation Times = <span className="font-bold">{head.actTimes}</span>)
            </span>
          </span>
          <Button onClick={() => setShowCustomInput((prev) => !prev)}>
            {showCustomInput ? "Hide Custom Input" : "Try Custom Input"}
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          {showCustomInput && <HeadCustomInputArea head={head} />}

          <div id="Activation" className="flex flex-col w-full gap-4">
            <HeadSampleGroup head={head} samples={head.samples} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
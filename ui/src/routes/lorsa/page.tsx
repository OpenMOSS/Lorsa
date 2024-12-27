import { AppNavbar } from "@/components/app/navbar";
import { LorsaHeadCard } from "@/components/lorsa/lorsa-vis";
import { SectionNavigator } from "@/components/app/section-navigator";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { HeadSchema } from "@/types/head";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useAsyncFn, useMount } from "react-use";
import { z } from "zod";

export const LorsaPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const [lorsaState, fetchLorsa] = useAsyncFn(async () => {
    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/lorsas`)
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  const [selectedLorsa, setSelectedLorsa] = useState<string | null>(null);

  const [headIndex, setHeadIndex] = useState<number>(0);
  const [loadingRandomHead, setLoadingRandomHead] = useState<boolean>(false);

  const [headState, fetchHead] = useAsyncFn(
    async (lorsa: string | null, headIndex: number | string = "random") => {
      if (!lorsa) {
        alert("Please select a lorsa head first");
        return;
      }

      setLoadingRandomHead(headIndex === "random");

      const head = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/lorsas/${lorsa}/heads/${headIndex}`,
        {
          method: "GET",
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
            stopPaths: ["sample_groups.samples.context"],
          })
        )
        .then((res) => HeadSchema.parse(res));
        setHeadIndex(head.headIndex);
      setSearchParams({
        lorsa,
        headIndex: head.headIndex.toString(),
      });
      return head;
    }
  );

  useMount(async () => {
    await fetchLorsa();
    if (searchParams.get("lorsa")) {
      setSelectedLorsa(searchParams.get("lorsa"));
    }
    if (searchParams.get("headIndex")) {
      setHeadIndex(parseInt(searchParams.get("headIndex")!));
      fetchHead(searchParams.get("lorsa"), searchParams.get("headIndex")!);
    }
  });

  useEffect(() => {
    if (lorsaState.value && selectedLorsa === null) {
      setSelectedLorsa(lorsaState.value[0]);
      fetchHead(lorsaState.value[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lorsaState.value]);

  const sections = [
    {
      title: "Histogram",
      id: "Histogram",
    },
    {
      title: "Logits",
      id: "Logits",
    },
    {
      title: "Top Activation",
      id: "Activation",
    },
  ].filter((section) => (headState.value && headState.value.logits != null) || section.id !== "Logits");

  return (
    <div id="Top">
      <AppNavbar />
      <div className="pt-4 pb-20 px-20 flex flex-col items-center gap-12">
        <div className="container grid grid-cols-[auto_600px_auto_auto] justify-center items-center gap-4">
          <span className="font-bold justify-self-end">Select Lorsa:</span>
          <Select
            disabled={lorsaState.loading || headState.loading}
            value={selectedLorsa || undefined}
            onValueChange={setSelectedLorsa}
          >
            <SelectTrigger className="bg-white">
              <SelectValue placeholder="Select a lorsa" />
            </SelectTrigger>
            <SelectContent>
              {lorsaState.value?.map((lorsa, i) => (
                <SelectItem key={i} value={lorsa}>
                  {lorsa}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            disabled={lorsaState.loading || headState.loading}
            onClick={async () => {
              await fetchHead(selectedLorsa);
            }}
          >
            Go
          </Button>
          <span className="font-bold"></span>
          <span className="font-bold justify-self-end">Choose a specific lorsa head:</span>
          <Input
            disabled={lorsaState.loading || selectedLorsa === null || headState.loading}
            id="head-input"
            className="bg-white"
            type="number"
            value={headIndex.toString()}
            onChange={(e) => setHeadIndex(parseInt(e.target.value))}
          />
          <Button
            disabled={lorsaState.loading || selectedLorsa === null || headState.loading}
            onClick={async () => await fetchHead(selectedLorsa, headIndex)}
          >
            Go
          </Button>
          <Button
            disabled={lorsaState.loading || selectedLorsa === null || headState.loading}
            onClick={async () => {
              await fetchHead(selectedLorsa);
            }}
          >
            Show Random Lorsa Head
          </Button>
        </div>

        {headState.loading && !loadingRandomHead && (
          <div>
            Loading Lorsa Head <span className="font-bold">#{headIndex}</span>...
          </div>
        )}
        {headState.loading && loadingRandomHead && <div>Loading Random Living Lorsa Head...</div>}
        {headState.error && <div className="text-red-500 font-bold">Error: {headState.error.message}</div>}
        {!headState.loading && headState.value && (
          <div className="flex gap-12">
            <LorsaHeadCard head={headState.value} />
            <SectionNavigator sections={sections} />
          </div>
        )}
      </div>
    </div>
  );
};

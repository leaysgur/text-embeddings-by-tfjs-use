import { For } from "solid-js";
import { createStore, produce } from "solid-js/store";
import { render } from "solid-js/web";
import { cosineSimilarity } from "./math";

const App = () => {
  const [state, setState] = createStore({
    use: null,
    entries: [],
    generating: false,
    results: [],
  });

  const loadTfjs = async (ev) => {
    ev.currentTarget.disabled = true;
    ev.currentTarget.textContent = "Loading...";

    console.log("USE loading...");
    console.time("USE loaded");
    const [, { load }] = await Promise.all([
      import("@tensorflow/tfjs"),
      import("@tensorflow-models/universal-sentence-encoder"),
    ]);
    const use = await load();
    console.timeEnd("USE loaded");

    setState(produce((draft) => (draft.use = use)));
    ev.currentTarget.textContent = "Loaded!";
  };

  const addEntry = () =>
    setState(produce((draft) => draft.entries.push({ embeddings: null })));
  const deleteEntry = (entryIdx) =>
    setState(
      produce(
        (draft) =>
          (draft.entries = draft.entries.filter((_, idx) => idx !== entryIdx))
      )
    );

  const generateEmbeddings = async (entryIdx, text) => {
    if (text.trim() === "") return;
    if (state.use === null) return;

    setState(
      produce((draft) => {
        draft.generating = true;
        draft.entries[entryIdx].embeddings = "Generating...";
      })
    );

    const tensor = await state.use.embed(text);
    const [embeddings] = await tensor.array();
    setState(
      produce((draft) => {
        draft.entries[entryIdx].embeddings = embeddings;
        draft.generating = false;
      })
    );
  };

  const calcDisabled = () => {
    if (state.generating) return true;
    if (state.entries.some((e) => !Array.isArray(e.embeddings))) return true;
    if (state.entries.filter((e) => Array.isArray(e.embeddings)).length < 2)
      return true;
    return false;
  };
  const calcCosineSimilarity = async () => {
    setState(produce((draft) => (draft.results = [])));

    const results = [];
    const header = [""];
    for (const [idx] of Object.entries(state.entries))
      header.push(`#${Number(idx) + 1}`);
    results.push(header);

    for (const [idx, e1] of Object.entries(state.entries)) {
      const row = [`#${Number(idx) + 1}`];
      for (const [, e2] of Object.entries(state.entries)) {
        const cosSim = cosineSimilarity(e1.embeddings, e2.embeddings);
        row.push(cosSim);
      }
      results.push(row);
    }

    setState(produce((draft) => (draft.results = results)));
  };

  return (
    <>
      <h1>Tensorflow.js USE Embeddings</h1>

      <fieldset disabled={state.use !== null} style="display: contents">
        <button onClick={loadTfjs}>Load tfjs and USE</button>
      </fieldset>
      {" | "}
      <fieldset disabled={state.use === null} style="display: contents">
        <button onClick={addEntry}>Add entry</button>
        <button onClick={calcCosineSimilarity} disabled={calcDisabled()}>
          Calc cosine similarity
        </button>
      </fieldset>

      <hr />

      <For each={state.entries}>
        {(entry, idx) => (
          <div style="display: grid; gap: 5px; grid-template-columns: max-content 1fr 1fr max-content">
            <span>#{idx() + 1}</span>
            <textarea
              placeholder="Enter sentence..."
              onBlur={(ev) => generateEmbeddings(idx(), ev.currentTarget.value)}
            />
            <textarea readOnly>{JSON.stringify(entry.embeddings)}</textarea>
            <button onClick={() => deleteEntry(idx())}>x</button>
          </div>
        )}
      </For>

      <hr />

      <table border="1" style="width: 100%">
        <tbody>
          <For each={state.results}>
            {(row) => (
              <tr>
                <For each={row}>
                  {(v) => (
                    <td
                      style={{
                        color: 0.5 < v ? "crimson" : "inherit",
                        ["background-color"]: 0.8 < v ? "pink" : "inherit",
                      }}
                    >
                      {v}
                    </td>
                  )}
                </For>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </>
  );
};

render(() => <App />, document.getElementById("app"));

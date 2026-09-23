const form = document.querySelector("#prediction-form");

const percent = (value) => `${(value * 100).toFixed(1)}%`;

async function requestPrediction(home, away) {
  const params = new URLSearchParams({ home, away });
  const response = await fetch(`/api/predict?${params}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "無法產生預測");
  return data;
}

function renderPrediction(data) {
  const home = data.home_team;
  const away = data.away_team;
  document.querySelector("#result-matchup").textContent = `${home} vs ${away}`;
  document.querySelector("#home-win-value").textContent = percent(data.probabilities.home_win);
  document.querySelector("#draw-value").textContent = percent(data.probabilities.draw);
  document.querySelector("#away-win-value").textContent = percent(data.probabilities.away_win);
  document.querySelector("#home-win-bar").style.width = percent(data.probabilities.home_win);
  document.querySelector("#draw-bar").style.width = percent(data.probabilities.draw);
  document.querySelector("#away-win-bar").style.width = percent(data.probabilities.away_win);
  document.querySelector("#home-xg-team").textContent = home;
  document.querySelector("#away-xg-team").textContent = away;
  document.querySelector("#home-xg").textContent = data.expected_goals.home.toFixed(2);
  document.querySelector("#away-xg").textContent = data.expected_goals.away.toFixed(2);
  document.querySelector("#likely-score").textContent = `${data.most_likely_score.home} : ${data.most_likely_score.away}`;
  document.querySelector("#top-scorelines").replaceChildren(
    ...data.top_scorelines.map((scoreline) => {
      const item = document.createElement("li");
      const score = document.createElement("strong");
      const probability = document.createElement("span");
      score.textContent = `${scoreline.home}–${scoreline.away}`;
      probability.textContent = percent(scoreline.probability);
      item.append(score, probability);
      return item;
    }),
  );
  document.querySelector("#model-factors").replaceChildren(
    ...data.factors.flatMap((factor) => {
      const term = document.createElement("dt");
      const detail = document.createElement("dd");
      term.textContent = factor.label;
      detail.textContent = factor.value;
      return [term, detail];
    }),
  );
}

async function loadPrediction(homeOverride, awayOverride) {
  const home = homeOverride || document.querySelector("#home-team").value;
  const away = awayOverride || document.querySelector("#away-team").value;
  const error = document.querySelector("#form-error");
  const result = document.querySelector("#result-card");

  error.hidden = true;
  result.setAttribute("aria-busy", "true");

  if (home === away) {
    error.textContent = "請選擇兩支不同球隊。";
    error.hidden = false;
    result.setAttribute("aria-busy", "false");
    throw new Error(error.textContent);
  }

  try {
    const data = await requestPrediction(home, away);
    renderPrediction(data);
    return data;
  } catch (caught) {
    error.textContent = caught.message;
    error.hidden = false;
    throw caught;
  } finally {
    result.setAttribute("aria-busy", "false");
  }
}

form?.addEventListener("submit", (event) => {
  event.preventDefault();
  loadPrediction().catch(() => {});
});

function registerPredictionTool() {
  const context = document.modelContext;
  if (!context?.registerTool || !form) return;

  const validTeams = new Set(
    Array.from(document.querySelectorAll("#home-team option"), (option) => option.value),
  );

  try {
    Promise.resolve(
      context.registerTool({
        name: "predict_epl_match",
        title: "預測英超賽果",
        description: "使用 EPL Predictor 的 Enhanced Dixon-Coles V2 模型預測指定主隊與客隊的 1X2 機率，並同步更新畫面。",
        inputSchema: {
          type: "object",
          properties: {
            home_team: { type: "string", description: "主隊的資料集名稱" },
            away_team: { type: "string", description: "客隊的資料集名稱" },
          },
          required: ["home_team", "away_team"],
          additionalProperties: false,
        },
        annotations: { readOnlyHint: true, untrustedContentHint: false },
        async execute(input) {
          const home = input?.home_team;
          const away = input?.away_team;
          if (!validTeams.has(home) || !validTeams.has(away)) {
            throw new Error("球隊名稱不在目前 EPL 資料集中。");
          }
          if (home === away) throw new Error("主隊與客隊必須不同。");

          document.querySelector("#home-team").value = home;
          document.querySelector("#away-team").value = away;
          const data = await loadPrediction(home, away);
          return {
            matchup: `${home} vs ${away}`,
            probabilities: data.probabilities,
            expected_goals: data.expected_goals,
            most_likely_score: data.most_likely_score,
            top_scorelines: data.top_scorelines,
            model: data.model,
            version: data.version,
            app_version: data.app_version,
            model_version: data.model_version,
          };
        },
      }),
    ).catch(() => {});
  } catch (_) {
    // The page remains fully usable in browsers without WebMCP support.
  }
}

registerPredictionTool();
loadPrediction().catch(() => {});

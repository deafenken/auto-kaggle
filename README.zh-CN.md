# Auto Kaggle Skills

`auto-kaggle` 是一套面向 Claude Code 和 Codex agent 的「给一个 Kaggle 比赛 URL → 自动多日刷奖牌」的分阶段 skill 集合。

整体思路:把刷 Kaggle 拆成 4 个阶段,每个阶段由一个独立 skill 负责,阶段之间通过 `runs/<comp_slug>/` 目录下的文件交接,**所有状态落盘**,**进程随时可崩可重启**,直到比赛 deadline 或者你放下一个 `STOP` 文件。

---

## 一句话理解

> 你扔过来一个 Kaggle 比赛 URL,它问你算力环境后,自动下数据 → 抓最热的公开 kernel 学技巧 → 自己用 CV-aware 流水线训练 → 实时显示「今日已用 3/5 提交,下次刷新还有 2h17m」 → 候选提交按 trust-adjusted CV 排序后**等你来挑**,挑完它真去 `kaggle competitions submit`,然后接着练下一个。

它**不是**自动作弊系统、**不是**自动投稿就完事、**不是**「无脑 fork 最高公开 kernel」工具。最终 2 个 final submission 由你按 deadline 时的推荐自己挑。

---

## 小白须知(第一次用之前请先看这里)

### 这个 skill 适合谁

- 已经有 Kaggle 账号、能登录、能在网页上接受比赛规则。
- 想稳一块银牌 / 冲一块金牌(默认 tier 就是 `silver-floor-gold-ceiling`)。
- 至少有一台机器能**长时间挂着**(本地 GPU、云 GPU、或者打开的 Kaggle Notebook),否则「全自动多日刷」就无从谈起。
- 你愿意每天看一两次 `recommendations.md`,在它列出的候选里挑一个让它提交。

### 这个 skill **不适合**谁

- 还没注册 Kaggle、不知道一个比赛长什么样的人——先去刷个 `titanic` 走通一遍流程。
- 想跑 Knowledge / Tutorial 类无奖牌比赛——bootstrap 阶段会直接拒。
- 想绕开提交配额、多账号刷、抄公开 kernel 不署原作者——integrity rules 里第 1/2/5 条直接挡住。

### 第一次使用前要准备的东西

1. **装好 Claude Code 或 Codex CLI**,且能跑通基本对话。
2. **装好 Kaggle CLI 并完成认证**:
   ```bash
   pip install --upgrade kaggle
   # 然后在 https://www.kaggle.com/<你的用户名>/account 下载 kaggle.json
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   kaggle competitions list   # 能列出比赛说明配好了
   ```
3. **在浏览器里登录 Kaggle,点开你要打的比赛,点 "Late Submission" 或 "Submit" 按钮接受规则**。这个步骤无法自动化,不接受规则,后面 `kaggle competitions download` 一定 403。
4. **准备好计算环境**(`bootstrap` 阶段会问你选哪个):
   - `kaggle-notebook`:免费 T4×2 / P100,9 小时单 kernel 上限,30h GPU/周。code-only 比赛必须走这条路。
   - `local-gpu`:本机 GPU,告诉 skill 型号和显存。
   - `cloud-gpu`:Colab Pro / Lambda / Vast / RunPod,你自己保证机器不掉。
   - `cpu-only`:只在小表格比赛上有意义。
5. **了解几个名词**:
   - **public LB / private LB**:公榜是测试集一小部分的实时打分;deadline 后才出私榜,奖牌按私榜定。**盲跟公榜会被 shake-up 洗出奖牌区**,所以本 skill 的最终选择必须 CV-first。
   - **shake-up**:public→private 名次大洗牌,公榜前 10 私榜 200+ 是日常。
   - **CV**(cross-validation):本地交叉验证分数,是你能不能稳住私榜的唯一信号。
   - **daily quota**:每天 5 次提交(部分比赛不同),00:00 UTC 重置。
   - **attribution**:在 submission message 里写明你借鉴了哪几个公开 kernel——这是 Kaggle 社区底线,本 skill 强制执行。
6. **先读 `auto-kaggle/references/integrity-rules.md`** 里的 10 条,**特别是第 1/3/4/5 条**,这 4 条决定了你账号会不会被举报/封号。

### 如果你是刚接触 Kaggle 的新手(最重要,先做完这步)

软件准备再齐全,不懂 Kaggle 文化也刷不出来。建议**先**:

1. **走通 Titanic 入门**:用 `auto-kaggle titanic` 跑一次完整流程,体会 4 个阶段、recommendations.md、提交配额这些是怎么联动的。这步不会掉奖牌(Titanic 没奖牌),只是熟悉机制。
2. **挑 1 个**「正在进行中、不那么热门」的比赛(参赛队 < 1000、deadline 还有 1 个月以上的 Playground Series 是不错的起点),先用 skill 全自动跑一周,观察:
   - CV / public LB 之间的 gap 是不是稳定
   - `recommendations.md` 推荐的 top 候选最后实际位次怎样
   - 你自己介入挑选 vs. skill 推荐第一名之间的差别有多大
3. 等你能用三五句话说清「这个比赛 top 公开 kernel 都在用什么招、private LB 大概率会怎么洗、我加什么能赢他们」,**再**去打一个真正想拿牌的比赛。

这一步比下面任何技术准备都重要。否则你只是在烧 GPU 时间和提交配额。

---

## 四个 Skill 各自负责什么

| Skill | 角色 | 输入 | 主要产物 |
|---|---|---|---|
| `auto-kaggle` | 总控(不做研究,只调度 + 守门) | 比赛 URL + 用户 4 个答案 | `runs/<slug>/run.yaml`,heartbeat / progress.jsonl,把守每次阶段交接 + 集成度规则 |
| `auto-kaggle-bootstrap` | Stage 0:解析比赛 + 下数据 + 问算力 | 比赛 URL | `comp_profile.yaml` / `rules_summary.md` / `data_stats.md` / `compute_env.yaml` |
| `auto-kaggle-recon` | Stage 1:定时抓 top 公开 kernel,提炼想法 | comp_profile + 已抓时间戳 | `kernels_index.json` / `ideas_pool.md` / `citations.bib` |
| `auto-kaggle-modeling` | Stage 2:自己写 pipeline,带 CV,把 recon 的想法落到自己代码里 | ideas_pool + comp_profile | `pipeline.py` / `runs/<run_id>/` / `leaderboard.csv` |
| `auto-kaggle-submit` | Stage 3:排候选、追配额、写推荐、提交、记 ledger | leaderboard + quota_state | `recommendations.md` / `submission_log.jsonl` / `quota_state.yaml` / `wait_until.txt` |

`auto-kaggle` 自己**不抓 kernel、不写模型、不提交**。它只调度这 4 个,守 integrity rules,管 resume / wait_until / heartbeat。

---

## 流程图

```
                  ┌──────────────────────────────────────────────────────────┐
                  │       auto-kaggle  (总控 / orchestrator)                 │
                  └──────────────────────────────────────────────────────────┘
                                       │
   首次? ──────yes──────►  Stage 0:auto-kaggle-bootstrap
                                       │   问算力 + 下数据 + 解析规则 + 探任务类型
                                       ▼
                       ┌── 定时 recon ───►  Stage 1:auto-kaggle-recon
                       │                      抓 top 公开 kernel,提炼 ideas(带引用)
                       │                      每 N 小时一次,可配置
                       ▼
                  Stage 2:auto-kaggle-modeling
                     自己 pipeline + CV-aware 训练
                     把 recon 的 ideas 一个一个塞进去,每个都有 ablation
                       │
                       ▼
                  Stage 3:auto-kaggle-submit
                     按 trust-adjusted CV 排候选 → 检查配额 → 写 recommendations.md
                     等你来挑 → 真去提交 → 记 submission_log.jsonl
                       │
                  配额烧完? ─yes─► 写 wait_until.txt → skill 退出
                       │ no
                       └──► 接着 modeling / recon / submit 循环
```

---

## 为什么要折腾「长时间不掉」

Kaggle 比赛跨**天到月**,而 agent 进程很容易因为以下原因停掉:

- Claude Code 的 context 满了被你 `/clear`
- 你关了笔记本电脑
- 网断了 `kaggle competitions submit` 失败
- 机器重启
- token 用完账户暂停

skill 设计上对这些都是**可恢复**的,因为:

- **所有状态在磁盘上**:`.heartbeat`、`progress.jsonl`、`submission_log.jsonl`、`quota_state.yaml`、`wait_until.txt` 全是文件,没有 agent 跨调用的记忆。
- **每个微步骤都写一行 `progress.jsonl`**:崩了之后 resume 第一件事就是读这个文件的最后一行,接着干。
- **append-only ledger**:`submission_log.jsonl` 永不重写,半截写入也只是丢最后一行。
- **wait_until 协议**:配额烧完不死等,写文件 + 退出,外层调度器读这个时间睡觉。
- **supervisor 脚本** `auto-kaggle/assets/supervisor.sh`:外面套个 `while true` + Claude/Codex headless,崩了自动拉起,直到 deadline 或你放 `STOP` 文件。

3 种调度方式,根据你能挂多久选:

- **`manual`**:你每天有空的时候来一句 `/auto-kaggle resume <slug>`,最稳。
- **`claude-loop`**:Claude Code 里 `/loop /auto-kaggle resume <slug>`,Claude Code 开着就会自己跑。
- **`shell-supervisor`**:`nohup bash auto-kaggle/assets/supervisor.sh <slug> > supervisor.log 2>&1 &`,在一台 24h 开着的机器上 fire-and-forget,直到 deadline。

任何时刻你想停:
```
touch runs/<comp_slug>/STOP
```
任何时刻你想暂停:
```
touch runs/<comp_slug>/PAUSE     # 删掉文件就恢复
```
任何时刻你想看进度而不打扰它:
```
cat runs/<comp_slug>/.heartbeat
tail -n 20 runs/<comp_slug>/progress.jsonl
cat runs/<comp_slug>/stage3_submit/recommendations.md
cat runs/<comp_slug>/stage3_submit/quota_state.yaml
```

---

## 设计原则(integrity rules,详见 `auto-kaggle/references/integrity-rules.md`)

1. **No verbatim copying** —— 公开 kernel 只作参考,代码必须自己重写 + 引用源 kernel。
2. **每次 submit message 强制带 `attr:`** —— 引用前 1–3 个最影响这次提交的 kernel。
3. **CV-first selection** —— 永远不按 public LB 排名挑最终提交。
4. **No LB probing** —— 近重复提交会被本地 block。
5. **单账号** —— 多账号刷一律拒绝。
6. **配额诚实** —— 不允许提交一堆乱数烧配额。
7. **CV 不能被反向调** —— 调 split 去贴 LB 是私榜炸的捷径。
8. **算力预算闸** —— 跑超预算前先警告。
9. **deadline 模式** —— deadline 前 24h 切换到「只 ensemble、不试新架构、最后 2 选必须用户点头」。
10. **外部数据必须 Kaggle-shared** —— 否则可能 DQ。

---

## 安装

### Claude Code

```bash
mkdir -p ~/.claude/skills
cp -r auto-kaggle auto-kaggle-bootstrap auto-kaggle-recon \
      auto-kaggle-modeling auto-kaggle-submit ~/.claude/skills/
```
或者放进项目级 `<project>/.claude/skills/`。然后 `/skills` 确认 5 个名字都在。

### Codex / OpenAI 兼容 agent

把同样 5 个目录放进 Codex skills 目录即可,`agents/openai.yaml` 提供 UI 元数据。

---

## 快速开始

```text
你:auto kaggle https://www.kaggle.com/competitions/playground-series-s4e5

agent:
  → 问你 4 个问题:算力环境 / 用户名 / 目标段位 / 调度模式
  → Stage 0:下数据、解析规则、写 comp_profile.yaml,告诉你 deadline 还有 27 天
  → Stage 1:抓 top 30 公开 kernel,提炼 14 个 ideas
  → Stage 2:跑一个 5-fold LightGBM baseline,CV RMSE 0.745
  → Stage 3:写 recommendations.md,列出当前唯一候选,quota 0/5,等你确认
你:submit candidate 1
agent:
  → kaggle competitions submit ...,public LB 0.748
  → 接着跑下一组 ideas...
  → 当日 5/5 用完:写 wait_until.txt = 2026-05-13T00:00:00Z,退出
  (supervisor 睡到 UTC 0 点把它叫醒,quota 重置,继续)
```

随时:
```
cat runs/playground-series-s4e5/.heartbeat
cat runs/playground-series-s4e5/stage3_submit/recommendations.md
```

---

## 仓库结构

```text
auto-kaggle/             # 总控
auto-kaggle-bootstrap/   # Stage 0
auto-kaggle-recon/       # Stage 1
auto-kaggle-modeling/    # Stage 2
auto-kaggle-submit/      # Stage 3
README.md
README.en.md
README.zh-CN.md
```

每个 skill 目录:
- `SKILL.md`:触发 + 工作流
- `references/`:按需加载的参考文档
- `assets/`:helper 脚本、模板、supervisor.sh
- `agents/openai.yaml`:Codex 端 UI 元数据,Claude 忽略

---

## 注意事项

- 这是辅助你刷 Kaggle 的基础设施,**不是免责声明**:账号该封还是会封。
- 最终 2 个 final submission 永远是你自己挑,skill 只排序 + 提建议。
- skill **不会**主动改 `cv_split.yaml`——CV 方案一旦定下,只能你改。
- 数据落在 `runs/<comp_slug>/data/`,这是 `.gitignore` 的,绝对不要 commit 进任何 repo。
- `kaggle.json` / 任何凭据**不要**放进 `runs/`,统一在 `~/.kaggle/` 里。

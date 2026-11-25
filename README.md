# A 股日频量化系统

用 SQLite 保存逐日行情和时点化输入。每个交易日收盘后先合并多路召回，再输出按综合分排序的候选股。训练、每日预测和回测都通过命令行执行。建议用 Python 3.11 或 3.12。

## 安装和直接验证

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[futu]'
```

已有 NumPy、pandas、PyTorch 时可在项目目录直接运行 `python3 -m ashare`。只用 Tushare 时可安装 `python -m pip install -e .`。

```bash
python -m ashare --db data/demo.sqlite demo-data
python -m ashare --db data/demo.sqlite train --epochs 2 --out runs/demo-model
python -m ashare --db data/demo.sqlite predict --model runs/demo-model \
  --probabilities --out runs/demo-signals
python -m ashare --db data/demo.sqlite backtest --model runs/demo-model \
  --compare-baseline --out runs/demo-backtest
python -m unittest discover -s tests -v
```

模拟数据独立标记为 `synthetic`，假日按工作日简化，产生的收益与模型指标只用于功能验证。每次训练或研究运行使用新输出目录，保留原有版本。

## 本地行情网站与 Qlib 档案

下载同一个 GitHub 发布版本的两个全量包，保留原文件名。

- [2026-11-23 日线主包](https://github.com/xu-duqing/investment_data/releases/download/2026-11-23/qlib_bin.tar.gz)
- [2026-11-23 每日指标包](https://github.com/xu-duqing/investment_data/releases/download/2026-11-23/daily_basic_qlib_features.tar.gz)

```bash
python -m ashare import-qlib ~/Downloads/qlib_bin.tar.gz \
  ~/Downloads/daily_basic_qlib_features.tar.gz --release 2026-11-23 \
  --out data/qlib.sqlite --extract-to data/qlib-extracted
python -m ashare serve
```

打开 <http://127.0.0.1:8765/>。默认同时发现 `data/qlib.sqlite` 和 `data/ashare.sqlite`。可按交易日浏览截面、搜索证券和日期范围、查看 K 线与成交量、导出当前证券的历史 CSV，以及核对各年和各字段的覆盖。模拟库需明确用 `--db data/demo.sqlite` 选择。

`import-qlib` 对已核实版本校验压缩包 SHA-256，并校验两个包共享的交易日历和指标文件清单。SQLite 保存所有可读日频字段、交易日历、证券名单和指数成分区间，并计算有报价行、有效市值行和 OHLC 异常。缺值保留为空，两个包里均无任何有效字段的占位日不产生行情行。原始字段沿用 Qlib 口径，浏览器额外把每日指标总市值从万元换算到元。Qlib 的复权归一化价格和成交量与交易库的人民币价格、原始股数不同，Qlib 数据存入独立档案库。原有训练、策略回测命令仍读取原始行情库。

`--extract-to` 用于保留解包目录，省略时仅保存完整数据库。换版本时用新的输出路径。数据网站仅监听本机。

## Qlib 量价联合 token 训练

```bash
.venv/bin/python -m ashare --threads 5 train-joint --out runs/joint-token-v1
.venv/bin/python -m ashare --threads 5 train-joint --out runs/joint-token-markov-v1 \
  --markov-residual --epochs 4 --train-blocks 8192 --val-days 12 --test-days 24 \
  --exclude-tested runs/joint-token-v1/model.json
```

默认使用 `data/qlib.sqlite`。标签按交易日划分为 2022–2024 年训练、2025 年验证、2026 年测试。可用 `--end` 固定数据库的截止日期，`--test-days 0` 评估所有可用且未排除的测试交易日。

每个收盘日使用三个特征。总市值以人民币计并取对数。原始成交量用 Qlib 的 `volume × factor` 还原，再除以此前 20 个交易日的成交量中位数后取对数，至少需要 15 个有效交易日。复权收益用相邻交易日复权归一化收盘价计算。无报价、停牌间断或缺失市值会切断上下文。分位点和收益桶代表值只用训练期拟合。三个特征各分 10 桶，组合成 1,000 个 token。

模型含 5 层因果 Transformer、32 维隐藏层、4 个注意力头、24 个交易日上下文。联合 token embedding 与三个分量的 embedding 相加。输出层预测下一交易日联合 token，按价格桶求边际概率，再按桶内训练期均值加权得到期望收益。训练用联合交叉熵与收益桶交叉熵。可用 `--markov-residual` 在 logits 中加入训练期一阶 Markov 先验，让网络学习其修正量。

训练对可用盘面抽取时序块。验证与测试按交易日均匀取样，并评估入选日期的全部合格证券。以验证收益桶 NLL 选择权重，独立报告测试指标，并对比无条件分布和一阶 Markov。每次运行输出 `weights.pt`、`model.json` 和逐日 `test_rank_ic.csv`，可用 `ashare.joint_token.load_joint_checkpoint` 校验和加载模型。样本外指标衡量预测和排序表现，成交约束与交易成本需另做回测。

## 5,000 token 与训练页

```bash
.venv/bin/python -m ashare --threads 5 train-typed --out runs/typed-token-v1
.venv/bin/python -m ashare serve
```

打开 <http://127.0.0.1:8765/training> 查看运行状态、每轮训练与验证耗时、四项 loss、各批次耗时、验证集 Rank IC 和测试集模型对照。训练最多隔五个批次原子更新进度。每次训练需要使用一个新的运行目录。

三项量价特征分十桶并与五个股票类型组合，共 5,000 个联合 token。五类是主板、科创板、创业板、北交所、ST 或疑似 ST。六层因果 Transformer 用两个输出头组成联合分布，并用独立排序头预测横截面强弱。新增 20 日波动率、5 日相对收益、市值前值年龄、ST 与涨跌停来源等连续或标记特征。行业成员信息按生效和可知时间使用，缺失时以同板块收益为参照。市值缺失时最多沿用此前三个交易日，并保留年龄。

本地原始行情库的官方涨跌停数据优先。可用 `--status-csv` 扩充，表头 `day,code,is_st,limit_up,limit_down`，日期为交易日，`is_st` 取 0 或 1 或留空，涨跌停价用人民币元。同一证券日一行。无官方状态时，板块根据代码划分，主板在历史 5% 限幅时期用已有多次封板特征标记疑似 ST。2026 年 7 月 6 日及以后停止用 5% 规则识别主板 ST。推算限价跳过新股前五个交易日、复权因子变化和明显超出边界的日期。

预测特征截至信号日收盘。概率模型预测下一交易日 token，排序目标用下一交易日开盘到再下一交易日开盘的复权收益。开盘到达涨停价、缺报价或限价无法判定的标的排除在排序样本外，并保留在有合法 token 的概率任务中。排序损失比较同日相对最强与最弱组，给强组中超过样本中位数更多的股票最高两倍权重。采用验证集排序 Rank IC 选权重，一阶和三阶 Markov 都只用训练期，三阶按低阶平滑回退。出场价格、队列和成本需要独立交易回测。

## Tushare 与一年日线

在项目目录设置 `TUSHARE_TOKEN`，也可把 Token 保存为仅当前用户可读的 `data/tushare.token`，文件只写 Token 本身。程序通过官方 HTTPS API 发送认证。数据目录和运行结果目录不属于发布包。

```bash
# 单日连通性和权限测试
python -m ashare sync --start 2026-11-23 --end 2026-11-23
# 最近一年和 120 个自然日预热。后续执行会接着未完成的日期同步
python -m ashare sync
python -m ashare status

# 历史行业回测优先导入有存档的时点记录
python -m ashare import-industries path/to/industries.csv
# 实时行业召回可用以下命令，按下载时间记录该快照
python -m ashare sync-industries

python -m ashare train --out runs/ts-v1
python -m ashare backtest --model runs/ts-v1 --compare-baseline --out runs/research-v1
python -m ashare predict --model runs/ts-v1 --probabilities --out runs/signals-v1
```

同步用 `trade_cal` 管理交易日，用 `stock_basic` 的上市、退市及暂停上市名单建立股票池，然后按交易日合并 `daily`、`daily_basic`、`adj_factor`、`stk_limit`。成交量统一为股，成交额和总市值统一为元。可用 `--rpm` 按账户权限限流。回补窗口相邻时会补全中间日期，失败日期在重跑时续传。北交所覆盖取决于供应商返回，逐日对预计上市代码做覆盖检查。实际有成交的股票如果缺市值或复权因子，当日同步报错。缺涨跌停信息的股票可用于信号研究，但当日不会模拟成交。

申万一级行业也可用 `sync-industries --assume-effective-known` 从生效日重建历史分类。这是一项回测假设，原始发布或收到时间仍应优先。新闻通过 `import-hotspots` 接入已标注的行业或股票事件，新闻接口及主题标签生产需要另行接入。字段模板见 `examples/`。持仓可用 `import-holdings path/to/holdings.csv --asof 2026-11-23T16:00:00+08:00` 导入，空快照可表示清仓。每次预测只读取信号截点前的最新一份持仓快照。

官方接口文档：[HTTPS](https://tushare.pro/document/1?doc_id=130)、[日线](https://tushare.pro/document/1?doc_id=27)、[每日指标](https://tushare.pro/document/2?doc_id=32)、[复权](https://tushare.pro/document/2?doc_id=28)、[涨跌停](https://tushare.pro/document/2?doc_id=183)、[行业历史成分](https://tushare.pro/document/1?doc_id=335)。

## 富途 OpenD

启动并登录 OpenD 后，在同一台电脑的普通终端运行。以下命令用贵州茅台、平安银行试拉过去一年的原始和后复权日线。省略起止日期时，默认多下载 120 个自然日供预热。可以用 `--codes path/to/codes.txt` 提交股票池，每行一个富途代码。下载按 OpenD 历史额度分批进行，同一下载目录可续传。

```bash
python -m ashare futu-pull --code SH.600519 --code SZ.000001 \
  --start 2025-11-23 --end 2026-11-23 --out data/futu-20261123
```

历史 K 线原始价格、成交量和成交额可以直接复用。历史总市值、乘法复权因子、各交易日上下限需从逐日参考数据合并。缺失限价时 `limit_known=0`，确定当日无涨跌幅限制时两列留空且 `limit_known=1`。下载的后复权日 K 单独留存和交叉检查，合并采用逐日参考表的复权因子。

```bash
python -m ashare --db data/futu.sqlite futu-import \
  --pull data/futu-20261123 --fundamentals path/to/fundamentals.csv \
  --securities path/to/securities.csv
```

`fundamentals.csv` 列为 `day,code,market_cap,adj,limit_up,limit_down,limit_known`。`securities.csv` 列为 `code,name,listed,delisted`。内部代码格式为 `600519.SH`、`000001.SZ`、`xxxxxx.BJ`。富途适配器拉沪深行情，数据库按对应股票池绑定来源。常规 `import-market DIR --source SOURCE` 也可接入外部数据。目标目录需含 `securities.csv`、`sessions.csv`（`day` 列）、`bars.csv`（见 `ashare.store.BAR_FIELDS`，同一交易日连续排列）。

官方接口文档：[历史日 K](https://openapi.futunn.com/futu-api-doc/quote/request-history-kline.html)、[市场快照](https://openapi.futunn.com/futu-api-doc/quote/get-market-snapshot.html)、[历史额度](https://openapi.futunn.com/futu-api-doc/intro/authority.html)。

## 策略与模型

| 组件 | 当前做法 |
| --- | --- |
| 热门行业及新闻 | 用行业收益、上涨广度、相对成交额计算热度，再召回行业内成交活跃股。合并截点前可用的行业和个股新闻事件 |
| 成交量异动 | 当日成交量除以此前 20 个有成交日的中位量。默认达到 1.8 倍并满足最低成交额 |
| 持仓召回 | 预测读取已导入的持仓快照。回测使用模拟账户当时的持仓。停牌持仓出现在候选表，缺有效分时无名次 |
| 趋势 35% | 5 日、20 日复权收益加权 |
| MACD 25% | 归一化的 12/26/9 MACD 柱 |
| 时序模型 40% | 用最近 20 个连续交易日的收益率、对数总市值、对数成交量和相对量 Token，预测下一交易日的复权收盘涨跌幅分桶概率 |
| 综合分 | 每个模型先按当日候选股做横截面百分位，再加权。计算可用权重覆盖率，覆盖不足的股票保留明细 |

训练仅用训练段拟合各字段分桶边界。下一日标签为下一交易日复权收盘相对当日复权收盘的收益率。训练样本的标签分桶均值为各桶代表涨跌幅，时序原始分是 `Σ P(桶) × 桶代表涨跌幅`。训练、验证和测试按标签所在交易日切分，默认 65% / 15% / 20%，也可指定 `--train-end` 和 `--val-end`。验证对数损失决定模型版本，留出期报告对数损失、准确率、收益误差和每日横截面 Rank IC，并与训练标签的无条件分布比较。模型回测的首个收盘信号不能早于模型验证截止日。

参数放在 `examples/strategy.json`，用 `--config` 加载。默认权重属于初始设定。独立训练评估与组合基线用于判断模型是否增益。回测执行以次日开盘价格为代理，时序分预测的是收盘到收盘，跨隔夜方向信号和实际可交易收益可能不同。

## 回测和输出

### 交易参考

```bash
python -m ashare sync-trade-reference
python -m ashare serve --port 8798
```

打开 [交易参考](http://127.0.0.1:8798/trade)。页面使用最近可用的收盘数据，显示候选池中得分最高的 30 只股票，每只附代码、名称、综合分、分项分数、收盘价和召回来源。默认使用 35% 原版低波和 65% MLP（量价＋估值），可以切换到其他低波组合。综合分采用与低波增强回测相同的截面分位加权。

`sync-trade-reference` 检查 GitHub 最新发布，只下载相互配对的日线和每日指标时间增量包。它校验 SHA-256、发布基线和交易日历，更新独立的 `data/qlib-trade.sqlite`，并原子替换 `data/trade-reference.json`。历史回测默认继续使用 `data/qlib.sqlite`。漏掉发布时，用 `--release YYYY-MM-DD` 按发布顺序补齐。GitHub API 限流时，可以传入 `--release-metadata data/releases/2026-11-24/release.json` 读取同目录的已下载配对增量。也可用 `rank-trade-reference` 仅从本地数据重新评分。若指定了自定义排名输出文件，给 `serve` 传入同一 `--trade-reference` 路径。

### Top 10 买入与 Top 20 保留

```bash
python -m ashare --threads 5 backtest-rotation --out runs/rotation-factor-comparison-v2
python -m ashare serve --port 8768
```

打开 [组合回测](http://127.0.0.1:8768/backtest) 查看净值、回撤、因子对照、同期基准和成交明细。此命令使用本地 Qlib 和已完成训练的 `runs/typed-token-v1`，默认从 2026 年第一个交易日开始。支持 `--start`、`--end`、`--cash`、`--model` 与 `--status-csv`。

年度下拉提供 2024、2025、2026 三个独立账户回放。每年从 100 万元空仓开始，2026 年止于行情库最后一日。切换年份会保留已选排序方案，便于同因子比较。

成交记录和下载的 CSV 按含交易所后缀的股票代码关联证券简称。默认使用本地证券库及 Qlib 数据库旁的 `security-names.csv`。该表的基础清单来自 [A 股基础数据](https://github.com/whz06/deep-learning2026final-lab/tree/main/train_li/data)，缺项由腾讯行情补齐。名称用于展示，历史更名以成交日的公告为准。

含 Transformer 的回测使用各年度对应的模型。2024 版训练期为 2020–2022 年、验证期为 2023 年。2025 版训练期为 2021–2023 年、验证期为 2024 年。2026 版训练期为 2022–2024 年、验证期为 2025 年。各版在验证结束后固定权重，用于对应年度回放。复现 2024 年回放的命令如下。

```bash
python -m ashare --threads 3 train-typed --start 2020-01-01 --train-end 2022-12-31 --val-end 2023-12-31 --end 2024-12-31 --test-days 12 --out runs/typed-token-2024
python -m ashare --threads 3 backtest-rotation --start 2024-01-01 --end 2024-12-31 --model runs/typed-token-2024 --out runs/rotation-factor-2024-v2
```

每日收盘将已有持仓加入候选，然后依次补入历史行业、新闻、成交量异动和按成交额补齐的股票。候选上限 1,000 只，选股分数由趋势 45%、MACD 35% 与 Transformer 排序头 20% 加权。收盘后生成交易信号，次日开盘先卖出跌出前 20 的持仓，再以净值的 10% 为每笔新仓的预算，从前 10 只中的非持仓股票补位。无有效排名而无法判断的持仓保留待查。

按开盘价成交，佣金双边各 3 bp、最低 5 元，双边过户费各 0.1 bp，卖出印花税 5 bp。开盘限价检查使用官方限价或前收盘与历史状态推算的限价。涨跌停、缺价、手数及信号日成交量都会影响成交，存量仓位随市值变化。对照包括趋势、MACD、Transformer、成交量异动、5 日反转、20 日低波动六个单因子，以及四个多因子组合。

指数基准默认读取 `data/benchmark-index-closes.csv`，可用 `--benchmark-csv` 替换。CSV 列为 `day,code,close,source`，代码为沪深 300（000300）、科创综指（000680）、科创 50（000688）。同期收益以首个信号日与期末收盘点位计算，逐日曲线需要完整交易日数据。页面同时提供 Qlib 全市场及科创板股票等权参考。

### 固定数量等权回测

- 信号在北京时间 17:30 截面生成，按当日收盘价定目标股数，下一交易日开盘模拟成交。默认等权 12 只，日频调仓，允许持有现金。策略文件可调整调仓频率。
- 模拟检查停牌、开盘涨跌停、手数和可用资金。用信号日成交量估计可成交股数。买入股数和卖出股数都只在开盘处理一次，符合 T+1 的日频约束。费率、滑点、最小佣金按配置扣减。
- 收益曲线逐日以收盘价估值。分红、送股和拆合股用复权因子变化近似。长期停牌沿用最后价格，已知退市后按零回收估值。开盘撮合采用先卖后买的简化，真实成交队列、精确账户分红与冲击成本需要逐笔或公司行动数据。
- `backtest --compare-baseline` 用同一输入、时段和执行假设运行规则组合。输出资金/净值、交易、被阻订单、候选排名、公司行动代理事件和汇总指标。附全样本有连续报价的股票等权收益参考序列。
- `predict` 输出候选排名、召回证据、每个模型原始分和百分位、权重覆盖率及数据来源。加 `--probabilities` 可导出时序分桶概率，分桶定义在 `model.json`。预测命令只生成信号文件。

### 低波增强研究

```bash
python -m ashare --threads 5 train-low-vol --out runs/low-vol-2020-2023-v1
python -m ashare --threads 2 backtest-rotation --low-vol-model runs/low-vol-2020-2023-v1 \
  --start 2024-01-01 --end 2024-12-31 --out runs/rotation-low-vol-2024
```

原版 20 日低波保留 65% 排序权重，增强规则或模型占 35%。低波特征使用至多 60 个历史交易日。5 日均线日斜率是相邻均线值之比取对数，模型用平均斜率、斜率标准差、正斜率比例和以标准差调整的斜率描述上涨的方向和稳定性。其余特征包含波动、下行波动、回撤、动能和规模。估值对照加入正市净率与正市盈率的倒数、股息率和缺失标记。

使用 `--low-vol-grid` 可对同一批固定模型测试 65%、50%、35% 原低波权重，以及三种卖出规则。两日规则仅在连续两个交易日均有有效排名且排名和大于 50 时触发。增强分缺失时使用原低波排名。纯低波的三种卖出规则也单独列为对照，合计 48 个组合。

```bash
python -m ashare --threads 2 backtest-rotation --low-vol-model runs/low-vol-2020-2023-v1 --low-vol-grid \
  --start 2024-01-01 --end 2024-12-31 --out runs/rotation-low-vol-grid-2024
```

回测页在 2024、2025、2026 年的权重与卖出对照记录齐备时，按 `(1 + R2024) × (1 + R2025) × (1 + R2026) − 1` 排序，并列出各年的净收益和最大回撤。各年独立空仓起算，2026 年只算到本地行情的最后一个交易日，表中显示截止日期。

模型用次日开盘到第六个交易日开盘的可成交收益截面排名训练。2020–2022 年先拟合，2023 年用来选择 MLP 训练轮次，然后用截至 2023 年末的合规标签重训 Ridge 和 MLP。回测使用固定权重，放量召回和换仓约束与原版相同。

### 低波策略日内买卖

```bash
python -m ashare --threads 2 backtest-rotation --low-vol-model runs/low-vol-2020-2023-v1 --intraday \
  --start 2026-01-01 --end 2026-12-31 --out runs/rotation-intraday-2026
```

固定 35% 原版低波和 65% 量价＋估值 MLP。比较开盘基线、单改买入、单改卖出、双侧限价顺位和双侧限价固定买单。两日排名和大于 50 的持仓开盘卖出，启用提前止盈的方案在排名和大于 30 且不超过 50 时提交昨收上方 1% 的卖单。启用限价买入时，按昨收下方 1% 挂单。买价向下、卖价向上取整到分。开盘越过挂价时取开盘价，其余用日 K 触价判断。有复权因子变化或缺少可比较昨收时跳过条件单。

顺位版按全天是否触价依次检查 Top 30，日 K 无法判断真实触价顺序和挂单排队。固定买单版在开盘后按空位锁定 Top 10 中的买单和资金，当天不再补位。两种规则都不使用日内提前止盈释放的资金和空位补买。明细列出成交规则、昨收、挂单价及未触价原因。

```bash
python -m ashare --threads 2 backtest-rotation --low-vol-model runs/low-vol-2020-2023-v1 --intraday \
  --slippage-bps 10 --compare-slippage --start 2024-01-01 --end 2026-11-23 \
  --out runs/rotation-slippage-2024-2026
```

`--slippage-bps` 设置单边滑点。买价上调、卖价下调，开盘单受交易所涨跌停价限制，日内限价单还受挂单价限制。以无滑点参考价和模拟成交价的差额乘成交股数，单列滑点成本。仓位、现金和手续费均按模拟成交价计算。`--compare-slippage` 生成每个方案的 0 bp 对照。该连续回测跨年延续持仓。

回测页的季度与年度收益从逐日净值计算。首期使用回测首日收盘，其余各期使用上期末收盘，并以本期最后一个可用收盘日为终点。所有指数使用完全相同的起止交易日。缺少端点给出空值，超额收益用策略净收益减指数收益。同期基准还显示策略与指数同日收益率的皮尔逊相关系数。

红利低波用中证指数官网日收盘点位。H20269 计入分红再投资，H30269 反映价格变化。回测通过复权因子近似处理公司行动，未单独模拟股息税。页面的增强模型对照沿用 2024、2025、2026 年度权重与卖出回测，各年独立空仓，统一按开盘价调仓、两日排名和大于 50 卖出，滑点为 0 bp。估值版同时加入账面市值比、盈利收益率和股息率，收益差反映这组特征一起加入后的结果。

## 代码分层

| 层 | 模块 |
| --- | --- |
| 行情适配与时点输入 | `providers.py`、`futu_feed.py`、`store.py`、`schema.sql` |
| 统一因子计算 | `features.py` |
| 召回与组合打分 | `rank.py`、`models.py`、`config.py` |
| 时序训练、模型文件、概率推理 | `temporal.py` |
| 账户、撮合和绩效 | `backtest.py` |
| 同步、训练、推理、回测入口 | `cli.py` |

SQLite 按 `(day, code)` 存行情，`(code, day)` 索引用于个股历史。每个交易日的数据在一个事务提交。模型目录保存权重、权重校验码、训练/验证/测试区间、分桶配置和评估指标。新增自训模型可实现 `rank.ScoreModel` 的 `score(day, candidates, features)`，在 `models.ARTIFACT_LOADERS` 注册加载器，然后在策略 JSON 写模型权重。回测按自训模型声明的验证截止点约束可用日期。当前用固定模型进行留出期回测，可通过不同训练窗口重复运行得到滚动研究。

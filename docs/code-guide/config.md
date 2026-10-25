# config.py：传达室大爷——先把配置单填好再开工

对应源码：[agent_demo/config.py](D:/study/AI/ai-agent-demo-study/agent_demo/config.py) · [返回导航](D:/study/AI/ai-agent-demo-study/docs/code-guide/README.md)

## 一句话定位

程序开工前要问一堆事：模型的密钥是啥？数据库在哪台机器？我叫什么用户 ID？历史看几轮？这些都填在项目根目录的 `.env` 文件里。

`config.py` 就是**传达室大爷**：去 `.env` 把信息收上来、转换类型，装进一个叫 `Settings` 的"配置袋子"。创建时自动检查范围、必填项、地址和端口，通过后才发给后面所有模块。它不连数据库、不问模型，只负责读配置和检查写法。

为什么要这么干？防止每个文件各自去翻 `.env`，有人读法还不一样。集中一处，出错一处报。

## 它的工作流程（先记住这张图）

```text
.env 文件 / 系统环境变量
    → load_settings() 把文字读出来
    → 数字文字转成数字（"3306" → 3306）
    → 装进 Settings 袋子，自动执行 __post_init__()
        → 先检查用户 ID、记忆窗口和检索范围
        → 最后调用 self.validate()，检查必填项、地址和端口
    → 把袋子交给 app.py / evaluate.py
    → AgentApp 使用检查通过的配置，创建模型、数据库和知识库
```

## 0. 开头那行 ROOT：从自己脚下认路

```python
ROOT = Path(__file__).resolve().parents[1]
```

一步步读：

| 写法 | 白话 |
| --- | --- |
| `__file__` | "我自己"这个 config.py 文件在哪 |
| `Path(...)` | 把路径字符串变成"路径专用工具对象" |
| `.resolve()` | 整理成完整绝对路径 |
| `.parents[0]` | 往上第一层 = `agent_demo` 目录 |
| `.parents[1]` | 往上第二层 = 项目根目录 |

于是 `ROOT / ".env"` 就是根目录那份 `.env`（`/` 在这里是**拼路径**，不是除法）。这样不管你终端在哪个目录启动程序，大爷都能找到自家的 `.env`，不会跑到别的文件夹乱找。

## 1. Settings：配置袋子长什么样

```python
@dataclass(frozen=True)
class Settings:
    llm_api_key: str = field(default="", repr=False)
    user_id: str = "local_user"
    summary_keep_turns: int = 6
    ...
```

| 写法 | 白话 |
| --- | --- |
| `@dataclass` | "自动生成模板"贴纸：帮你把创建对象、逐字段赋值、打印这些体力活全生成好，不用手写 `__init__` 里几十行 `self.xxx = xxx` |
| `frozen=True` | **袋子封口**。创建好之后 `settings.user_id = "别人"` 会直接报错。防止程序跑一半被谁偷偷改了配置 |
| `summary_keep_turns: int = 6` | 这一格叫 summary_keep_turns，预期整数，默认 6；仅决定压缩时保留多少轮原文 |
| `field(default="", repr=False)` | 默认空着；打印整个袋子时**隐藏这格**——密钥不会被意外显示在日志里 |

注意 `repr=False` 只是"打印时遮住"，**不是加密**。`.env` 里密钥本身还是明文，别把项目文件夹随便外传。

袋子里的格子分七组：

| 组 | 字段 | 回答什么问题 |
| --- | --- | --- |
| 模型 | `llm_api_key`、`llm_base_url`、`llm_model`、`embedding_model` | 找谁干活、地址、凭据 |
| 数据库 | `mysql_host` / `port` / `user` / `password` / `database` | 去哪台 MySQL、用哪个库 |
| 用户 | `user_id` | 聊天和画像归谁 |
| 摘要 | `context_token_budget`、`summary_keep_turns` | 单次主问答总 token 超过多少才摘要、压缩时保留多少轮原文 |
| 检索 | `rag_top_k`、`rag_max_distance` | 最多捞几段资料、多不相关就扔 |
| 回答检查 | `reflection_enabled` | 要不要交卷前检查一次，有问题再改一次 |
| 目录 | `data_dir`、`docs_dir` | 运行数据和预置资料放哪 |

`data_dir`、`docs_dir` 写死在代码默认值里，`.env` 没给它们开入口——想改只能改代码（评测代码就是用"另造一份 Settings"的方式换目录的）。

## 2. `__post_init__()`：装完袋子自动摇一摇

这个方法**不用任何人调用**——`@dataclass` 约定：字段刚装完，自动执行它。相当于袋子的"出厂自检"。

### 自检一：用户 ID 格式

```python
if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.user_id):
    raise ValueError("USER_ID 只能包含 1～64 个字母、数字、下划线或连字符")
```

那串 `r"..."` 是**正则表达式**，翻译成人话：

- `[A-Za-z0-9_-]`：每个字符只能是字母、数字、下划线或连字符；
- `{1,64}`：总共 1 到 64 个；
- `fullmatch`：**整个字符串**都要符合（不是"沾到一点就行"的 match）；
- 前面 `not`：不符合就进报错分支。

为什么管这么严？user_id 将来要进 SQL 查询、进文件路径。现在卡住格式，后面所有用到它的地方都能偷懒。

`raise ValueError(...)` = 举手喊"这份配置有问题！"，程序停止正常流程，把这句话一路往上扔。

### 自检二：数字范围

```python
if min(self.context_token_budget, self.summary_keep_turns, self.rag_top_k) < 1:
    raise ValueError("摘要阈值、原文保留轮数和检索数量必须大于 0")
if not 0 <= self.rag_max_distance <= 2:
    raise ValueError("RAG_MAX_DISTANCE 应在 0～2 之间")
```

`min(...)` 取三个数里最小的——**最小的都小于 1 了，说明肯定有不合格的**，一行顶三行 if。距离阈值限定 0～2 是因为 cosine 距离的取值范围就在这（0=一模一样，2=截然相反）。

函数签名 `-> None` 表示"检查完不交回东西，只负责喊对或喊错"。

### 最后一步：接着检查必填项、地址和端口

```python
self.validate()
```

这行在 `__post_init__()` 的最后。前面的范围检查通过了，就继续执行下面的 `validate()` 方法。`self` 表示“刚创建的这份配置”，所以检查的正是刚填进去的那些值。

`__post_init__()` 是 dataclass 自动调用的；`validate()` 是普通方法，由这一行接着调用。于是整个顺序是：**字段赋值 → 检查范围 → 检查必填项、地址和端口 → 交回配置对象**。其中任何一步抛出 `ValueError`，这次创建就失败，调用者拿不到配置对象。

## 3. `validate()`：继续验一遍必填项和连接参数

这里把较长的检查步骤单独放进一个方法，方便顺着读。调用时机已经统一到 `__post_init__()` 里，`AgentApp` 不再重复调用它。

```python
required = {
    "LLM_API_KEY": self.llm_api_key,
    "LLM_BASE_URL": self.llm_base_url,
    ...（共 7 项必填）
}
missing = [name for name, value in required.items() if not value.strip()]
if missing:
    raise ValueError("请在 .env 中填写：" + "、".join(missing))
```

- `required.items()`：遍历时一次拿到"配置名 + 值"这一对；
- `[name for ... if ...]`：列表推导式，读作“逐个检查，只收符合条件的名字”；
- `value.strip()`：去掉首尾空格再判断——只填了个空格的密钥也算没填；
- 不合格的名字全攒进 `missing` 列表，最后 `"、".join(missing)` 一次性全报给用户："请在 .env 中填写：LLM_API_KEY、MYSQL_PASSWORD"——**攒齐了一起说，不是发现一个就停**，让你改一遍 `.env` 就能全修好；
- 报错内容只有配置名，**绝不打印值**，密钥不进错误日志。

然后验地址和端口：

```python
url = urlparse(self.llm_base_url)
if url.scheme not in {"http", "https"} or not url.hostname:
    raise ValueError("LLM_BASE_URL 必须是有效的 API 地址")
if not 1 <= self.mysql_port <= 65535:
    raise ValueError("MYSQL_PORT 必须在 1～65535 之间")
```

`urlparse()` 把 `https://xxx.com/v1` 拆成协议、主机名等零件，检查协议是 http/https、主机名不是空。端口 1～65535 是 TCP 协议的合法范围。

**重要的认知边界：这只是检查"写法对不对"**。地址写得再漂亮，也不能证明服务在线、密钥有效、模型有权限——那些要等真发请求时才见分晓。

## 4. `load_settings()`：把全套串起来

```python
load_dotenv(ROOT / ".env", override=False, encoding="utf-8")
env = os.environ
```

`load_dotenv()` 干的事：读 `.env` 文件，把每一行"键=值"**塞进当前进程的 `os.environ`**（环境变量表）。它直接修改环境表，所以不需要接返回值，下一行 `env = os.environ` 就能读到。

`override=False` 是**优先级规则**：如果系统里已经有一个同名环境变量（比如 PowerShell 里 `$env:USER_ID="test"` 设过），就保留系统的，`.env` 只负责"补没有的"。方便你临时用环境变量覆盖配置调试，不用改文件。

```python
mysql_port=int(env.get("MYSQL_PORT", "3306")),
rag_max_distance=float(env.get("RAG_MAX_DISTANCE", "0.8")),
```

`env.get("名字", "默认值")`：环境里没设过这个键，就用默认文字。然后 `int()`、`float()` 把文字变成真数字——**环境变量全是字符串，这一层必须手动转换**。你在 `.env` 写 `MYSQL_PORT=abc`，`int("abc")` 当场炸出 `ValueError`。

一个坑值得记住：`get` 只在**键不存在**时才用默认值。`.env` 里写了 `MYSQL_PASSWORD=`（存在但为空），读出来是空字符串 `""`，**不会**变成默认值——最后由 `validate()` 抓到"必填项没填"。

最后直接 `return Settings(...)`：创建时自动执行 `__post_init__()`，先检查基本范围，最后调用 `validate()`。全部通过后，`load_settings()` 才会返回配置对象；缺少必填项、地址格式不对或端口越界，会在这里立即报错。

这条规则同样适用于手工创建 `Settings(...)` 和评测里的 `replace(settings, ...)`：`replace` 会复制字段，再创建一份新配置，所以也会自动检查。直接写 `Settings()` 会因为默认密钥、地址和数据库账号密码为空而报错；测试代码使用 `tests/conftest.py` 提供的虚构配置。检查只看有没有填、格式是否合规，不会验证真实密钥和数据库密码是否可用。

### 多一个文字转开关的例子：`REFLECTION_ENABLED`

`Settings` 里的 `reflection_enabled: bool = False` 控制是否请检查员在交卷前看一遍答案。默认关闭，开启后每轮多一次检查请求，有问题才再多一次重写。

```python
reflection_enabled=env.get("REFLECTION_ENABLED", "false").strip().lower() in {"true", "1", "yes", "on"},
```

从左往右读：没填就取 `"false"` → 去首尾空格 → 转小写 → 看是不是集合里的某个开启写法。匹配才得到 `True`，其他文字得到 `False`；建议只写 `true` 或 `false`，拼错不会主动报错。

不能写 `bool("false")`：Python 只看字符串是否为空，这五个字母不为空，结果反而是 `True`。这里必须检查文字内容。

原有 [问答流程](D:/study/AI/ai-agent-demo-study/docs/code-guide/agent.md) 读取这个布尔值，再决定是否调用 [reflection.py](D:/study/AI/ai-agent-demo-study/docs/code-guide/reflection.md)。更改后同样要完整重启。

## 5. 走一遍例子：压缩时保留 2 轮

`.env` 写 `SUMMARY_KEEP_TURNS=2`，完整重启后，`load_settings()` 将字符串转成整数并交给 `Settings` 校验。

假设已有 8 轮未压缩问答，这轮回答完成后共有 9 轮。主 Agent 最后一次模型返回的 `total_tokens` 严格大于 `CONTEXT_TOKEN_BUDGET=50000` 时，程序总结前 7 轮，留下最近 2 轮原文。本轮回答属于留下的原文。

没有触发摘要时，仍发送当前会话全部尚未压缩的问答。下一轮原文会继续增长，2 轮不是永久窗口。

旧的 `MEMORY_TURNS` 和 `MEMORY_CHAR_BUDGET` 已不再读取，可从本机 `.env` 删除。摘要不受 `REFLECTION_ENABLED` 控制，也没有摘要长度配置。

## 本篇要带走的三句话

1. 全项目只有这一个地方读 `.env`，其他模块只接袋子，配置"一处进、处处用"；
2. 创建 `Settings` 就自动检查：`__post_init__` 先查范围，再调用 `validate` 查漏填、地址和端口，通过后才能交给应用使用；
3. `frozen=True` + `repr=False`：封口防乱改，遮密钥防打印——都是小成本的大安心。

继续读：[llm.py：一部装好密钥的电话](D:/study/AI/ai-agent-demo-study/docs/code-guide/llm.md)。

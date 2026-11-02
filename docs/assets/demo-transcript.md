# 本地 Demo 真实日志

执行 `uv run python -m agentic_search.chat`，通过标准输入发送的问题是「星尘咖啡馆的招牌饮品是什么？」和 `exit`。重定向输入不会回显，因此问题单独列在此处。下方为完整输出（包含 jieba 启动日志）：

```text
Agentic Search — 角色: manager | Web 搜索: 关闭(未配置 TAVILY_API_KEY,纯本地模式) | exit 退出

你: 助手: 
🔍 检索: 星尘咖啡馆的招牌饮品
Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\ADMIN\AppData\Local\Temp\jieba.cache
Loading model cost 0.503 seconds.
Prefix dict has been built successfully.
   命中: meeting.wav, membership.md, menu.md, operations.md, suppliers.pdf
根据资料，星尘咖啡馆的招牌饮品是**星尘拿铁**，售价为32元，采用双份浓缩咖啡、燕麦奶打底，顶部撒海盐焦糖碎制成。另一款特色饮品为月岩冷萃，售价28元，经过16小时冷泡，带有可可与柑橘尾韵。

来源: menu.md
—— 本轮引用: meeting.wav, membership.md, menu.md, operations.md, suppliers.pdf

你: ```

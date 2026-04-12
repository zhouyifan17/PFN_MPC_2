评估模块说明文档
1. evaluate_models.py：全景开环仿真评估 (Full-Episode Open-Loop)
📌 核心定位“模型的长期稳定性与可视化诊断器”该脚本测试的是模型在完全“蒙眼闭环”状态下的长跑能力。它在给定初始状态后，仅依靠外部控制序列和自身的上一步预测，连续推演整个 Episode（如 2000 步）。支持single_step和rolling两种预测方式。
⚙️ 关键功能与工作流统一数据管道：自动从 data/processed/ 加载通过 joblib 持久化保存的 scaler_X.pkl 和 scaler_Y.pkl，保证推理时的数据尺度与训练时 $100\%$ 严格对齐。


2. evaluate_rollout.py：自回归滚动前瞻评估 (MPC Horizon Rollout) 这个文件并不常用。
📌 核心定位“MPC 求解器的终极模拟沙盘”该脚本测试的是模型在真实 MPC 控制循环中的短跑能力。它完全复刻了 MPC “滚动优化、反馈校正”的底层哲学。
⚙️ 关键功能与工作流双分支记忆机制：主分支（更新记忆）：在每一个时间步 $t$，模型“睁开眼睛”读取真实的物理传感器数据，推进并更新准确的内部隐状态 (h, c)。滚动分支（前瞻预测）：以当前的 (h, c) 为起点，闭上眼睛向未来预测一个固定长度的窗口（如 rollout_steps=10），模拟 MPC 的预测视界 (Prediction Horizon)。


💡 运行流程
训练完成 -> 运行 evaluate_models.py。
看图诊断 -> 打开 results/ 目录，检查图表。如果红线完全偏离黑线，说明训练参数或数据尺度有问题，立即返回修改，无需进行下一步。
定量测试 -> 图像表现良好后，运行 evaluate_rollout.py --rollout_steps 10。
记录指标 -> 将终端输出的 Rollout MSE 记录在案。挑选此指标最低且阴影区间最合理的 DeepAR 模型，送入 CasADi 编写机会约束！
<script setup>
import { Right } from '@element-plus/icons-vue'
import method from '../mds/method.mdx';
</script>

<template>
  <div id="method" v-animate-onscroll="'animated fadeInUp'">
    <el-row justify="center">
        <el-col :xs="28" :sm="24" :md="20" :lg="16" :xl="12">
          <h1>Method</h1>
            <div class="caption-box">
                <el-image src="./method.png" class="caption-img" fit="cover" />
                <method class="caption-text" />
            </div>

            <!-- 跨策略适配说明：两类策略家族如何接入同一个 PCD 插件 -->
            <div class="compat-card">
              <div class="compat-title">A single plugin for both policy families</div>
              <div class="compat-diagram">
                <div class="family-chip f1">
                  <span class="family-kind">Autoregressive</span>
                  <span class="family-name">OpenVLA</span>
                </div>
                <div class="connector c1">
                  <span class="conn-label">native action probabilities</span>
                  <span class="conn-line"></span>
                </div>
                <div class="pcd-node">
                  <span class="pcd-name">PCD</span>
                  <span class="pcd-sub">training-free plugin</span>
                </div>
                <el-icon class="conv-arrow" :size="18"><Right /></el-icon>
                <div class="out-node">
                  <span class="out-name">Enhanced actions</span>
                  <span class="out-sub">at test time</span>
                </div>

                <div class="family-chip f2">
                  <span class="family-kind">Diffusion</span>
                  <span class="family-name">Octo · π₀</span>
                </div>
                <div class="connector c2">
                  <span class="conn-label">KDE-PM estimation</span>
                  <span class="conn-line"></span>
                </div>
              </div>
              <div class="compat-note">
                Autoregressive policies expose action probability distributions natively, while diffusion-based
                policies obtain them through KDE-based Probabilistic Modeling (KDE-PM). Either way, PCD plugs in
                at test time — no fine-tuning, no access to model weights.
              </div>
            </div>
        </el-col>
    </el-row>

  </div>
</template>

<style scoped>

.caption-img {
    margin: 20px 0;
    border-radius: 12px;
    border: 1px solid var(--pcd-border);
    box-shadow: var(--pcd-shadow-soft);
}

/* 跨策略适配说明卡 */
.compat-card {
  margin: 20px 0;
  padding: 20px 22px;
  border-radius: 16px;
  background: var(--pcd-surface);
  border: 1px solid var(--pcd-border);
}

.compat-title {
  font-family: 'Google Sans', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: var(--pcd-heading);
  margin-bottom: 18px;
  text-align: center;
}

/*
 * 流程网格（保证两行严格对齐）：
 * 列 1 = 策略家族，列 2 = 连接线，列 3 = PCD 节点，列 4 = 箭头，列 5 = 输出
 */
.compat-diagram {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr) auto auto minmax(0, 0.9fr);
  grid-template-areas:
    "f1 c1 pcd arr out"
    "f2 c2 pcd arr out";
  gap: 16px 12px;
  align-items: stretch;
}

.f1 { grid-area: f1; }
.f2 { grid-area: f2; }
.c1 { grid-area: c1; }
.c2 { grid-area: c2; }

.family-chip {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 10px 16px;
  border-radius: 12px;
  background: #ffffff;
  border: 1px solid var(--pcd-border);
  box-shadow: 0 2px 8px rgba(16, 24, 40, 0.05);
}

.family-kind {
  font-family: 'Google Sans', sans-serif;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: var(--pcd-muted);
}

.family-name {
  font-family: 'Google Sans', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: var(--pcd-heading);
}

/* 连接线：文字在上，线条在下（带箭头） */
.connector {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
}

.conn-label {
  font-family: 'Inter', 'Noto Sans', sans-serif;
  font-size: 11.5px;
  color: var(--pcd-muted);
  white-space: nowrap;
}

.conn-line {
  position: relative;
  width: 100%;
  height: 2px;
  border-radius: 2px;
  background: #b7c9e4;
}

.conn-line::after {
  content: "";
  position: absolute;
  right: -1px;
  top: 50%;
  transform: translateY(-50%);
  border-left: 7px solid #b7c9e4;
  border-top: 4px solid transparent;
  border-bottom: 4px solid transparent;
}

/* PCD 节点 */
.pcd-node {
  grid-area: pcd;
  align-self: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 14px 22px;
  border-radius: 14px;
  background: linear-gradient(150deg, var(--pcd-accent), #7c5cf0);
  box-shadow: 0 8px 24px rgba(50, 115, 220, 0.35);
}

.pcd-name {
  font-family: 'Google Sans', sans-serif;
  font-size: 22px;
  font-weight: 700;
  color: #ffffff;
  line-height: 1.2;
}

.pcd-sub {
  font-family: 'Inter', 'Noto Sans', sans-serif;
  font-size: 11.5px;
  color: rgba(255, 255, 255, 0.88);
}

/* 汇聚箭头 */
.conv-arrow {
  grid-area: arr;
  align-self: center;
  justify-self: center;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border-radius: 50%;
  background: rgba(50, 115, 220, 0.1);
  color: var(--pcd-accent);
}

/* 输出节点 */
.out-node {
  grid-area: out;
  align-self: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 12px 18px;
  border-radius: 12px;
  background: #edf3fc;
  border: 1px solid rgba(50, 115, 220, 0.35);
}

.out-name {
  font-family: 'Google Sans', sans-serif;
  font-size: 14px;
  font-weight: 700;
  color: #1d5fc2;
}

.out-sub {
  font-family: 'Inter', 'Noto Sans', sans-serif;
  font-size: 11.5px;
  color: var(--pcd-muted);
}

.compat-note {
  margin-top: 16px;
  font-family: 'Inter', 'Noto Sans', sans-serif;
  font-size: 13.5px;
  line-height: 1.6;
  color: var(--pcd-muted);
  text-align: center;
}

/* 手机端：纵向堆叠 */
@media (max-width: 768px) {
  .compat-diagram {
    grid-template-columns: minmax(0, 1fr);
    grid-template-areas:
      "f1"
      "c1"
      "f2"
      "c2"
      "pcd"
      "out";
    gap: 10px;
    justify-items: center;
  }

  .conv-arrow {
    display: none;
  }

  .conn-line {
    width: 64px;
  }

  .compat-note {
    text-align: justify;
  }
}

</style>

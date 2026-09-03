<script>
// public 目录资源需用 BASE_URL 拼接，避免部署到 /PCD/ 子路径后 404
const videoSrc = import.meta.env.BASE_URL + 'videos/overview.mp4'

export default {
  name: 'Video',
  data() {
    return {
      videoSrc,
    };
  },
};
</script>

<template>
  <div>
    <!-- 每个网站的视频的iframe可能不一致，最好在这里手动调整 -->
    <el-row justify="center">
      <el-col :xs="28" :sm="24" :md="20" :lg="16" :xl="12" >

        <!-- local -->
        <el-container class="video-container">
          <video autoplay controls loop muted :src="videoSrc"></video>
        </el-container>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>

.video-container {
  margin: 24px 0;
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--pcd-border);
  box-shadow: var(--pcd-shadow-soft);
}

iframe, video {
  aspect-ratio: 16 / 9;
  width: 100%;
  display: block;
}

</style>
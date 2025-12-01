<script>
export default {
  name: 'Video',
  data() {
    return {
      playerOptions: {
        autoplay: true,
        controls: true,
        loop: true,
        muted: true,
        sources: [
          {
            src: '/PCD.mp4',
            type: 'video/mp4'
          }
        ],
        plugins: {
          // 这里可以添加其他插件配置
        }
      }
    };
  },
  mounted() {
    // 如果需要在组件挂载后执行一些操作，可以在这里添加
  }
};
</script>

<template>
  <div>
    <!-- 每个网站的视频的iframe可能不一致，最好在这里手动调整 -->
    <el-row justify="center">
      <el-col :xs="28" :sm="24" :md="20" :lg="16" :xl="12" >

        <!-- local -->
        <el-container class="video-container">
          <video autoplay controls>
            <source src="/PCD.mp4" type="video/mp4">
          </video>
        </el-container>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>

.video-container{
  margin: 20px;
}

iframe, video {
  aspect-ratio: 16 / 9;
  width: 100%;
}

</style>
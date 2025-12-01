<script setup>
import motivation from '../mds/motivation.mdx';
</script>

<template>
  <div v-animate-onscroll="'animated fadeInUp'">
    <el-divider />

    <el-row justify="center">
        <el-col :xs="28" :sm="24" :md="20" :lg="16" :xl="12">
            <div class="caption-box">
                <el-image src="./motivation.png" class="caption-img" fit="cover" />
                <div class="caption-text">
                  <motivation />
                </div>
            </div>
        </el-col>
    </el-row>

  </div>
</template>

<style scoped>

.caption-box {
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 20px 0;
}

.caption-img {
  width: 150%;
  max-width: 400px;
}

.caption-text {
    margin-left: 50px;
}

</style>
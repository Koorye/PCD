<script>
import { Swiper, SwiperSlide} from 'swiper/vue';
import { Navigation, Pagination, Autoplay} from 'swiper/modules';
import 'swiper/css/bundle';

export default {
  components: {
      Swiper,
      SwiperSlide,
      Navigation,
      Pagination,
      Autoplay,
    },
  data() {
    return {
      modules: [
        Navigation,
        Pagination,
        Autoplay,
      ],
      // 选择要轮播的照片
      imagesSeen: [
        {
            "name": "Baseline: Pick Coke Can",
            "path": "./videos/main/simpler/baseline/pick_coke_can.gif",
        },
        {
            "name": "+Ours: Pick Coke Can",
            "path": "./videos/main/simpler/pcd/cut_pick_coke_can.gif",
        },
        {
            "name": "Baseline: Move Near",
            "path": "./videos/main/simpler/baseline/move_near.gif",
        },
        {
            "name": "+Ours: Move Near",
            "path": "./videos/main/simpler/pcd/cut_move_near.gif",
        },
        {
            "name": "Baseline: Carrot Plate",
            "path": "./videos/main/simpler/baseline/carrot_plate.gif",
        },
        {
            "name": "+Ours: Carrot Plate",
            "path": "./videos/main/simpler/pcd/cut_carrot_plate.gif",
        },
        {
            "name": "Baseline: Eggplant Basket",
            "path": "./videos/main/simpler/baseline/eggplant_basket.gif",
        },
        {
            "name": "+Ours: Eggplant Basket",
            "path": "./videos/main/simpler/pcd/cut_eggplant_basket.gif",
        }
      ],
      imagesUnseen: [
        {
            "name": "Baseline: Brightness",
            "path": "./videos/factor/simpler/baseline/brightness.gif",
        },
        {
            "name": "+Ours: Brightness",
            "path": "./videos/factor/simpler/pcd/cut_brightness.gif",
        },
        {
            "name": "Baseline: Spatial Relation",
            "path": "./videos/factor/simpler/baseline/spatial.gif",
        },
        {
            "name": "+Ours: Spatial Relation",
            "path": "./videos/factor/simpler/pcd/cut_spatial.gif",
        },
        {
            "name": "Baseline: Texture",
            "path": "./videos/factor/simpler/baseline/texture.gif",
        },
        {
            "name": "+Ours: Texture",
            "path": "./videos/factor/simpler/pcd/cut_texture.gif",
        },
        {
            "name": "Baseline: Texture",
            "path": "./videos/factor/simpler/baseline/texture2.gif",
        },
        {
            "name": "+Ours: Texture",
            "path": "./videos/factor/simpler/pcd/cut_texture2.gif",
        }
      ],
    }
  }
}
</script>

<template>
  <div v-animate-onscroll="'animated fadeInUp'">
    <el-row justify="center">
      <h1>Simulation Demonstrations (SIMPLER)</h1>
    </el-row>

    <el-row justify="center">
    <el-col :xs="28" :sm="24" :md="20" :lg="16" :xl="12" class="swiper-box">
      <!-- 设置轮播图：循环播放、首张图序号、响应式、导航和分页、自动播放 -->
      <swiper
        :loop="true"
        :slidesPerView="1"
        :slides-per-group="2"
        :breakpoints="{
          600: {
            slidesPerView: 4,
            spaceBetween: 20,
          },
        }"
        :modules="modules"
        :navigation="{ 
          hideOnClick:true,
        }"
        :pagination="{ 
          hideOnClick:true,
          clickable:true, 
          type:'bullets' 
        }"
        :autoplay="{ 
          delay:5000,
          disableOnInteraction:false,
          pauseOnMouseEnter:true,
        }"
        >
        <!-- <swiper-slide v-for="path in image_paths"> -->
          <!-- <el-image :src="path"/> -->
        <!-- </swiper-slide> -->
        <swiper-slide v-for="(image, index) in imagesSeen" :key="index">
          <div class="swiper-slide-text">{{ image.name }}</div>
          <el-image class="swiper-slide-img" :src="image.path" :alt="image.name" fit="contain" />
        </swiper-slide>
      </swiper>
      <swiper
        :loop="true"
        :slidesPerView="1"
        :slides-per-group="2"
        :breakpoints="{
          600: {
            slidesPerView: 4,
            spaceBetween: 20,
          },
        }"
        :modules="modules"
        :navigation="{ 
          hideOnClick:true,
        }"
        :pagination="{ 
          hideOnClick:true,
          clickable:true, 
          type:'bullets' 
        }"
        :autoplay="{ 
          delay:5000,
          disableOnInteraction:false,
          pauseOnMouseEnter:true,
        }"
        >
        <!-- <swiper-slide v-for="path in image_paths"> -->
          <!-- <el-image :src="path"/> -->
        <!-- </swiper-slide> -->
        <swiper-slide v-for="(image, index) in imagesUnseen" :key="index">
          <div class="swiper-slide-text">{{ image.name }}</div>
          <el-image class="swiper-slide-img" :src="image.path" :alt="image.name" fit="contain" />
        </swiper-slide>
      </swiper>
    </el-col>
  </el-row>
  </div>
</template>

<style>

/* 设置Swiper风格 */
.swiper-box {
  background-color: var(--pcd-surface);
  border: 1px solid var(--pcd-border);
  border-radius: 16px;
  padding: 18px 14px;
}

.swiper {
  --swiper-theme-color: var(--pcd-accent);
}

.swiper-slide-text {
  color: var(--pcd-text);
  padding: 5px;
  font-size: 15px;
  font-weight: 500;
  text-align: center;
}

.swiper-slide-img {
  width: 100%;
  border-radius: 12px;
}

/* 分页圆点颜色 */
.swiper-box .swiper-pagination-bullet-active {
  background: var(--pcd-accent);
}

</style>
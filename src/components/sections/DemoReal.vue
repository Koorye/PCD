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
            "name": "Baseline: Pick Ball",
            "path": "./videos/main/real/baseline/pick_ball_thumb.gif",
        },
        {
            "name": "+Ours: Pick Ball",
            "path": "./videos/main/real/pcd/pick_ball_thumb.gif",
        },
        {
            "name": "Baseline: Move Near",
            "path": "./videos/main/real/baseline/move_near_thumb.gif",
        },
        {
            "name": "+Ours: Move Near",
            "path": "./videos/main/real/pcd/move_near_thumb.gif",
        },
        {
            "name": "Baseline: Banana Plate",
            "path": "./videos/main/real/baseline/banana_plate_thumb.gif",
        },
        {
            "name": "+Ours: Banana Plate",
            "path": "./videos/main/real/pcd/banana_plate_thumb.gif",
        },
        {
            "name": "Baseline: Stack Cube",
            "path": "./videos/main/real/baseline/stack_cube_thumb.gif",
        },
        {
            "name": "+Ours: Stack Cube",
            "path": "./videos/main/real/pcd/stack_cube_thumb.gif",
        }
      ],
      imagesUnseen: [
        {
            "name": "Baseline: Distractors",
            "path": "./videos/factor/real/baseline/distractor_thumb.gif",
        },
        {
            "name": "+Ours: Distractors",
            "path": "./videos/factor/real/pcd/distractor_thumb.gif",
        },
        {
            "name": "Baseline: Brightness",
            "path": "./videos/factor/real/baseline/brightness_thumb.gif",
        },
        {
            "name": "+Ours: Brightness",
            "path": "./videos/factor/real/pcd/brightness_thumb.gif",
        },
        {
            "name": "Baseline: Spatial Relation",
            "path": "./videos/factor/real/baseline/spatial_thumb.gif",
        },
        {
            "name": "+Ours: Spatial Relation",
            "path": "./videos/factor/real/pcd/spatial_thumb.gif",
        },
        {
            "name": "Baseline: Table Texture",
            "path": "./videos/factor/real/baseline/texture_thumb.gif",
        },
        {
            "name": "+Ours: Table Texture",
            "path": "./videos/factor/real/pcd/texture_thumb.gif",
        }
      ],
    }
  }
}
</script>

<template>
  <el-row class="swiper-box2" justify="center" v-animate-onscroll="'animated fadeInUp'">
    <el-col :xs="32" :sm="28" :md="24" :lg="20" :xl="16">
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
</template>
  
<style>

.swiper-box2 {
  background-color: #eeeeee;
  padding-bottom: 20px;
}

/* 设置Swiper风格 */
.swiper {
  --swiper-theme-color: white;
}

.swiper-slide-text {
/*  background-color: rgba(0, 0, 0, 0.5);*/
  color: black;
  padding: 5px;
  font-size: 16px;
  text-align: center;
}

.swiper-slide-img {
  width: 100%;
  border-radius: 20px;
}

</style>
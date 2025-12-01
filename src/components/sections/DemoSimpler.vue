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
  <!-- <el-divider /> -->

  <el-row class="swiper-box" justify="center" v-animate-onscroll="'animated fadeInUp'">
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

/* 设置Swiper风格 */
.swiper-box {
  background-color: #eeeeee;
  padding-top: 20px;
}

.swiper {
  --swiper-theme-color: white;
}

.swiper-slide-text {
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
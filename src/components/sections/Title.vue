<script lang="ts" setup>

import { ElIcon } from 'element-plus'
import { Link, Document } from '@element-plus/icons-vue'

// logo地址，没有则置为""即可
const logo = ''

// 标题
const title = 'Policy Contrastive Decoding'

// 标题颜色
const title_color = '#000000'

// 标题补充，没有则置为''即可
const title_supp = 'for Robotic Foundation Models'

// 标题补充颜色
const title_supp_color = '#000000'

// 按钮颜色
const btn_color = '#000000'

// 作者清单（包含作者姓名、头像、主页、地址序号）
const authors = [
  {
    name: "Shihan Wu",
    homepage: "https://Koorye.github.io/",
    address_flag: "1*"
  },
  {
    name: "Ji Zhang",
    homepage: "https://github.com",
    address_flag: "2*"
  },
  {
    name: "Xu Luo",
    homepage: "https://github.com",
    address_flag: "1"
  },
  {
    name: "Junlin Xie",
    homepage: "https://github.com",
    address_flag: "1"
  },
  {
    name: "Jingkuan Song",
    homepage: "https://github.com",
    address_flag: "3"
  },
  {
    name: "Heng Tao Shen",
    homepage: "https://github.com",
    address_flag: "3"
  },
  {
    name: "Lianli Gao",
    homepage: "https://github.com",
    address_flag: "1†"
  },
]

// 地址清单（包含地址名称、头像、主页、地址序号）
const addresses = [
  {
    address_flag: "1",
    name: "University of Electronic Science and Technology of China",
    homepage: "https://github.com",
    // icon: "./icons/uestc.jpg"
  },
  {
    address_flag: "2",
    name: "Southwest Jiaotong University",
    homepage: "https://github.com",
    // icon: "./icons/sjtu.jpg"
  },
  {
    address_flag: "3",
    name: "Tongji University",
    homepage: "https://github.com",
    // icon: "./icons/tju.jpg"
  },
]

// 共一和通讯提示
const con_and_corresponding_author = "*Equal Contribution  †Corresponding author"

const news = [
  "🔥May 23, 2025: Our paper has been updated for better clarity and readability. The optimized version is now available on arXiv.",
  "🔥May 20, 2025: The paper is now available on arXiv."
]

// 强调内容
const emphases = [
]

// 提供引导资料链接
const buttons = [
  {
    disabled: false,
    name: "ArXiv",
    link: "https://arxiv.org/abs/2505.13255",
    component: Link,
  },
  {
    disabled: false,
    name: "PDF",
    link: "https://arxiv.org/pdf/2505.13255",
    component: Document,
  },
  {
    disabled: false,
    name: "Code",
    link: "https://github.com/Koorye/PCD",
    component: Link,
  }
]

</script>

<template>
  <div>

    <!-- 最新消息提示 -->
    <el-row justify="center">
      <el-col :span="24">
        <el-row class="news-box" justify="center" v-for="(item, index) in news">
          <el-alert v-if="index == 0" :title="item" type="error" />
          <el-alert v-if="index > 0" :title="item" type="info" />
        </el-row>
      </el-col>
    </el-row>

    <!-- 文章logo -->
    <el-row v-if="logo" justify="center">
      <el-image :src="logo" class="logo" fit="cover" />
    </el-row>

    <!-- 文章标题 -->
    <el-row justify="center">
      <el-col :span="20">
        <h1 class="paper-title">
          <div v-if="title" :style="{color:title_color}"> {{ title }}</div>
          <div v-if="title_supp" :style="{color:title_supp_color}"> {{ title_supp }}</div>
        </h1>
      </el-col>
    </el-row>

    <!-- 作者名单 -->
    <el-row justify="center">
      <a :href=author.homepage v-for="author in authors">
        <el-button class="title-button" type="primary" text>
          <span class="author">
            {{ author.name }}<sup v-if="author.address_flag" class="name_sup">{{ author.address_flag }}</sup>
          </span>
        </el-button>
      </a>
    </el-row>

    <!-- 地址名单 -->
    <el-row justify="center">
      <a :href=address.homepage v-for="address in addresses">
        <el-button class="title-button" type="primary" text>
          <el-avatar v-if="address.icon" :size="40" :src="address.icon" />
          <span class="address">
            <sup v-if="address.address_flag" class="address_sup">{{ address.address_flag }}</sup>{{ address.name }}
          </span>
        </el-button>
      </a>
    </el-row>

    <!-- 共一和通讯提示内容 -->
    <el-row justify="center" class="con-cor">
        {{ con_and_corresponding_author }}
    </el-row>

    <!-- 强调内容 -->
    <el-row justify="center" class="emphasis" v-for="emphasis in emphases">
        {{ emphasis }}
    </el-row>

    <!-- 提供引导按钮 -->
    <el-row justify="center" style="margin-bottom: 20px;">
      <el-col :span="20">
        <el-row justify="center">
          <a :href=button.link v-for="button in buttons">
            <el-button class="guidance-button" size="default" :color="btn_color" :disabled="button.disabled" round>
              <el-icon :size="18">
                <component :is="button.component" />
              </el-icon>
              <span class="btn-text">{{ button.name }}</span>
            </el-button>
          </a>
        </el-row>
      </el-col>
    </el-row>

  </div>
</template>

<style scoped>

.news-box {
  animation: fadeIn 0.5s ease-in-out;
  margin: 10px;
}

/* 文章标题字体、字间距、居中排布、字号 */
.paper-title {
  font-family: 'Google Sans', sans-serif;
  font-weight: 600;
  letter-spacing: 0px;
  font-size: 3rem;
  margin: 32px;
  text-align: center;
}

/* 姓名和地址按钮 */
.title-button {
  margin: 0px 0px;
  padding: 5px;
}

/* 姓名和地址按钮光标悬浮 */
.title-button:hover {
  margin: 0px 0px;
}

/* 引导材料按钮 */
.guidance-button {
  margin: 8px 5px;
  padding: 18px 15px;
}

/* 姓名属性 */
.author {
  font-family: 'Google Sans', sans-serif;
  font-size: 20px;
}

/* 姓名上标属性 */
.name_sup {
  font-family: 'Google Sans', sans-serif;
  color: #606266; 
}

/* 地址属性 */
.address {
  font-family: 'Google Sans', sans-serif;
  font-size:  20px;
}

/* 地址上标属性 */
.address_sup {
  font-family: 'Google Sans', sans-serif;
  color: #606266; 
}

/* 头像属性 */
.el-avatar {
  margin-right: 5px;
}

/* 共一和通讯文字属性 */
.con-cor {
  font-family: 'Google Sans', sans-serif;
  font-size: 16px;
  margin: 18px 0px;
  text-align: center;
  white-space: pre;
}

/* 强调信息属性 */
.emphasis {
  font-family: 'Google Sans', sans-serif;
  color: chocolate;
  font-weight: bold;
  margin: 8px;
  font-size: 22px;
  text-align: center;
}

/* 引导材料按钮文字属性 */
.btn-text {
  font-family: 'Google Sans', sans-serif;
  font-size: 16px;
  color: #ffffff;
}

.el-alert {
  margin: 10px 0 0;
}

.el-alert:first-child {
  margin: 0;
}

.logo {
  width: 150px; 
  height: 150px;
  border-radius: 50%;
  box-shadow: #ced3dc 0px 0px 3px 2px;
  margin-top: 40px;
}

/* 手机端链接样式处理 */
a:-webkit-any-link {
  text-decoration: none;
}

/* 取消鼠标焦点悬浮在链接上的颜色装饰 */
a:hover {
  color: inherit;
  border-bottom: none;
}

/* 链接装饰，取消下划线和链接颜色 */
a {
	text-decoration: None;
	color: inherit;
}

</style>
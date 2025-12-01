import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import VueAnimateOnScroll from 'vue3-animate-onscroll'
import VueVideoPlayer from 'vue-video-player'
import 'element-plus/dist/index.css'
import "@/assets/font/font.css"

import App from './App.vue'
const app = createApp(App)

app.use(ElementPlus)
app.use(VueAnimateOnScroll)
app.use(VueVideoPlayer)
app.mount('#app')
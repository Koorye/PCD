<script>
import Title from './sections/Title.vue'
import Abstract from './sections/Abstract.vue'
import Motivation from './sections/Motivation.vue'
import Method from './sections/Method.vue'
import ExperimentSimpler from './sections/ExperimentSimpler.vue'
import ExperimentReal from './sections/ExperimentReal.vue'
import ExperimentFactors from './sections/ExperimentFactors.vue'
import Poster from './sections/Poster.vue'
import Citation from './sections/Citation.vue'
import DemoSimpler from './sections/DemoSimpler.vue'
import DemoReal from './sections/DemoReal.vue'
import Video from './sections/Video.vue'
import Highlights from './sections/Highlights.vue'
import Footer from './sections/Footer.vue'

export default {
  components: {
    Title,
    Abstract,
    Motivation,
    Method,
    ExperimentSimpler,
    ExperimentReal,
    ExperimentFactors,
    Poster,
    Citation,
    DemoSimpler,
    DemoReal,
    Video,
    Highlights,
    Footer,
  }
}
</script>

<template>
  <Title/>
  <Highlights/>
  <Video/>
  <DemoReal/>
  <DemoSimpler/>
  <Poster/>
  <Abstract/>
  <Motivation/>
  <Method/>
  <ExperimentSimpler/>
  <ExperimentReal/>
  <ExperimentFactors/>
  <Citation/>
  <Footer/>
</template>
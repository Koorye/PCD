<script>
import Title from './sections/Title.vue'
import Abstract from './sections/Abstract.vue'
import Motivation from './sections/Motivation.vue'
import Method from './sections/Method.vue'
import ExperimentSimpler from './sections/ExperimentSimpler.vue'
import ExperimentReal from './sections/ExperimentReal.vue'
import ExperimentFactors from './sections/ExperimentFactors.vue'
import Citation from './sections/Citation.vue'
import DemoSimpler from './sections/DemoSimpler.vue'
import DemoReal from './sections/DemoReal.vue'
import Video from './sections/Video.vue'

export default {
  components: {
    Title,
    Abstract,
    Motivation,
    Method,
    ExperimentSimpler,
    ExperimentReal,
    ExperimentFactors,
    Citation,
    DemoSimpler,
    DemoReal,
    Video,
  }
}
</script>

<template>
  <Title/>
  <Video/>
  <DemoReal/>
  <DemoSimpler/>
  <Abstract/>
  <Motivation/>
  <Method/>
  <ExperimentSimpler/>
  <ExperimentReal/>
  <ExperimentFactors/>
  <Citation/>
</template>
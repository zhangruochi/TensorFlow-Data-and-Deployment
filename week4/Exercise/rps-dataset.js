class RPSDataset {
  constructor() {
    this.labels = []
  }

  addExample(example, label) {
    if (this.xs == null) {
      this.xs = tf.keep(example);
      this.labels.push(label);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));
      this.labels.push(label);
      oldX.dispose();
    }
  }
  
  encodeLabels(numClasses) {
    for (var i = 0; i < this.labels.length; i++) {
      if (this.ys == null) {
        this.ys = tf.keep(tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)}));
      } else {
        const y = tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)});
        const oldY = this.ys;
        this.ys = tf.keep(oldY.concat(y, 0));
        oldY.dispose();
        y.dispose();
      }
    }
  }

  // splitTrainTest(ratio){

  //   var size = this.xs.shape[0];
  //   var test_size = parseInt(size * ratio);
  //   var train_size = size - test_size;
  //   const [train_xs, test_xs] = tf.split(this.xs, [train_size, test_size], 0);
  //   const [train_ys, test_ys] = tf.split(this.ys, [train_size, test_size], 0);

  //   console.log(train_xs.shape);
  //   console.log(train_ys);

  //   console.log(test_xs.shape);
  //   console.log(test_ys);

  //   return {train_xs: train_xs, 
  //       test_xs: test_xs, 
  //       train_ys: train_ys, 
  //       test_ys: test_ys}
  // }

}

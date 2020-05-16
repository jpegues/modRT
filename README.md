# Exploring Modifications for Decreasing the Runtime of a Ray-Tracer

Welcome!

The goal of this project was to use techniques of surface-level interpolation and computational geometry to significantly speed up the standard ray-tracing approach, all while still producing accurate final images.  We tested our modifications on a (simple) ray-tracer, and (long story short) we accomplished our goal!

Here's a quick example from our tests of how our MRT performs for a (very, very simple) underlying ray-tracer.  On the left, we have the image produced by a (simple) standard ray-tracer.  In the middle, we have the approximate image produced by our MRT.  On the right, we have the error (i.e., the difference in RGB color between the two images, divided by 3).  We expect the same/better performance for more advanced ray-tracers.

![Figure](https://github.com/jpegues/modRT/blob/master/example.png?raw=true)

Check out our report ("report_MRT.pdf") for a full description of our MRT algorithm, as well as performance analysis and discussion of the MRT compared to the standard ray-tracing approach.  Our report also contains possible avenues for improving the MRT algorithm, as well as suggestions for implementing the MRT algorithm on top of much fancier, more advanced ray-tracers out in the wild.  See our coding example ("example_MRT.ipynb") for a notebook that walks through how to use our implementation of the MRT algorithm.  Finally, see our coding tests ("test_MRT.ipynb") to reproduce all test cases discussed in the report.

This work was completed as the final project for a semester-long independent graduate research class.  While our project is over, we'd really love it if others tried these techniques out on more advanced, complex ray-tracers.  If you do, give us a shoutout, and let us know how the MRT algorithm performs!

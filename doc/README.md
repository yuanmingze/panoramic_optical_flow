# 1. Dataset

## 1.1. Generation

- [ ] Cross Image Boundary

## 1.2. TODO

- [ ] GT optical flow Occlusion
- [ ] FIx Poles optical flow


# 2. Method 

There are two steps:

**Method 1**: Warp based method

**Method 2**: CNN based method (CVPR 2021)
- [ ] Unsupervised Training, with warped image MISS /
  - Learning Stereo from Single Images
  - Learning optical flow from still images
- [ ] Change Choen's FFT to DCT, to solve the rotation invariance
- [ ] Kernel transform NN, 
- [ ] GNN the get the alignment coefficients;

## 2.1. Method 1

<p align="center"><img src="./images/method_1_pipeline.svg" align=middle alt="Icosahedron" style="width:1200px;"/></p>
A pipeline image of method; 

### 2.1.1. Image Alignment


### 2.1.2. Sub-Image Optical Flow
Geodesic polyhedron.


### 2.1.3. Sub-Image Optical Flow Blending


#### 2.1.3.1. Traditional Method

Generate the weight function:


#### 2.1.3.2. CNN Method

The CNN structure, and how to train.


### 2.1.4. Panoramic Image Optical Flow Generation

- [ ] Explain the 360 optical flow the shader;
- [ ] Fix the rendering code and generate a 360 dataset;
- [ ] Implement the code and test on Replica dataset;

### 2.1.5. Experiment

- [ ] Compare with others methods;
- [ ] How rotation affects accuracy?

### 2.1.6. Limitation

### 2.1.7. Conclusion



## 2.2. Geodesic Subdivision 

Principal polyhedron triangle are the seeds of ,
The face of the principal polyhedron is call principal polyhedron triangle (PPT).
The PPT face is subdivided by 

Loop Subdivision:

1. http://www.neolithicsphere.com/geodesica/doc/subdivision_classes.htm
2. https://en.wikipedia.org/wiki/Geodesic_polyhedron
3. http://pibeta.phys.virginia.edu/docs/publications/ketevi_diss/node34.html
4. https://graphics.stanford.edu/~mdfisher/subdivision.html


5. implement: https://www.opengl.org.ru/docs/pg/0208.html
6. implement: http://blog.coredumping.com/subdivision-of-icosahedrons/


7. Reference software: http://www.neolithicsphere.com/geodesica/index.htm


## 2.3. Gnomonic Projection

Azimuthal equidistant projection

https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection

https://fr.maplesoft.com/applications/view.aspx?SID=3583&view=html

https://casa.nrao.edu/aips2_docs/memos/107/node2.html

https://casa.nrao.edu/aips2_docs/memos/107/node2.html#SECTION00021100000000000000

http://www.geography.hunter.cuny.edu/~jochen/GTECH361/lectures/lecture04/concepts/Map%20coordinate%20systems/Perspective.htm


## 2.4. Reference

Reference: https://medium.com/@deviparikh/planning-paper-writing-553f497e8839

# 3. Code


## 3.1. Folder structure

```
.
├── paper_draft: latex file;
├── paper_draft/image: images for documents;
├── optical_flow_evaluation.md: the document of how to evaluate performance of optical flow.
├── README.md
```

`paper_draft` folder is paper's draft;
`tech_detail` is including the code technical detial;
`paper_plane` is the plane of the paper.


## 3.2. Data Convention

### 3.2.1. Numpy shape

The image and flow data store in Numpy array.
Image array shape is [height, width, 3], and optical flow array shape is [height, width, 2].

### 3.2.2. Coordinate System Convention

In this project the default coordinate system same as the following image.
- x,y and z are the cartesian coordinate system notation.
- φ and θ are the spherical coordinate system notation, which are the longitude and latitude respectively.
- u and v are the the image coordinate system notation, which are the column and row index respectively.

![coordinate System](./doc/images/coordinate_system.svg)

**Cartesian coordinate system (3D)**

The +X is right, +Y is down and +Z is forward (Right hand coordinate system), and the origin [0,0,0] is overlap with the spherical coordinate system.

**Gnomonic coordinate system (Tangent Image)**

The tangent image show in the image as a pink square.
It's generated with gnomonic projection (xy) is the gnomonic plane coordinate system (normalized coordinate) whose origin is image center, +x is right, + y is up. And the tangent point is the origin of image.

And the uv are the gnomonic image coordinate system whose origin is image top-left.

About the gnomonic projection please reference https://mathworld.wolfram.com/GnomonicProjection.html

**Spherical coordinate system**

Its origin is overlap with cartesian coordinate system.
The θ axis range is [-π, +π) which is consistent with xz plane.
And φ axis range is [-0.5 * π, + 0.5 * π) which is consistent with yz plane.

This project also use (longitude, latitude) spherical coordinate system notationm which is same as (theta, phi)

This project use a special convention introduces in Jump.
https://developers.google.com/vr/jump/rendering-ods-content.pdf

**Equirectangular Image**

The ERP image pixels coordinate origin is Top-Left, and the image row and column index are v and u, is in range [0, width) and [0, hight) respectively.

And the ERP image's spherical coordinate origin at image center, phi (latitude) and theta(theta) is (0,0).
The first pixel 0 is corresponding azimuth -π, and the last pixel image_width - 1 is corresponding +π.
The theta is [-π, +π), phi is [-0.5*π, +0.5*π].

### 3.2.3. Optical flow

The optical flow U is corresponding theta, and V is corresponding theta.
The optical flow data structure layers order are U and V.

### 3.2.4. Wrap around 

The default, the ERP optical flow wrap around. 
For example, if a pixel from the [0.8𝜋,0] cross the theta [+𝜋,-] to [-0.9𝜋,0], the pixel's optical flow is +0.3𝜋 not 1.7𝜋.
The project name this optical flow as ERP optical flow.
And another type of optical flow as Non-ERP optical flow.

# 4. Dataset Generation 

CNN Method
Geodesic Grids
Tangent Image
Spherical Coordinate system.

\subsubsection{Gnomonic Projection}

\textbf{Sphere 2 Perspective Image}

%Set the point $S$ as the centre point of the projection, who longitude and latitude is $(\theta, \phi)$

For a projection with the central point $(\theta_0, \phi_0)$,
\begin{itemize}
	\item Ohttps://mathworld.wolfram.com/GnomonicProjection.html
	\item Another entry in the list
\end{itemize}
\subsection{Panoramic Image Processing}

\subsection{Synthesize Panoramic Optical Flow}\label{sec:app:panoof}

The real word data rich and colorful, but it is very hard to estimate or measure the accurate segmentation, motion flow, e.t.c form the real word scene.
So the synthetic dataset is commonly haired to generate the ground truth data for training or performance evaluation, such as ~\cite{habitat19iccv}.

Meanwhile, the real word photo image is hard to estimate accurate optical flow.
Although KITTI~\cite{Menze2018JPRS} or MPI Sintel~\cite{Butler:ECCV:2012} pinhole image optical flow, e.t.c currently don't have any public panoramic optical flow dataset is available.

\begin{figure}[hbt!]
	\centering
	\includegraphics[width=\linewidth]{images/synthetic_optical_flow/of_render.pdf}
	\caption{The pipeline of panoramic optical flow rendering.}
	\label{fig:approach:panoof:pipline}
\end{figure}

The synthetic panoramic optical flow rendered with on-the-shelf OpenGL render pipeline and store in the ERP image.
The input are textured 3D mesh and OpenGL's camera pose.
The process shown as the Fig.~\ref{fig:approach:panoof:pipline} composing with the following 3 steps:

\begin{enumerate}
	\item Camera Model: OpenGl render with Equirectangular projection (ERP);
	\item Warp Around: Processing the warp around at the boundary of image;
	\item Occlusion: estimate the occlusion of optical flow ;
\end{enumerate}

\subsubsection{Camera Model}

The traditional OpenGL rendering pipeline doesn't support the panoramic camera model and optical flow generation.
For synthesising ground truth panoramic RGB images and optical flow, we implement and hire equirectangular perspective camera model.

The equirectangular perspective panoramic camera model render the 3D mesh to  equirectangular images.
To achieve the camera model the OpenGL geometry shaders transform the 3D mesh from Cartesian coordinate system to spherical coordinate system. 
The camera mode

Furthermore, we use Replica to demonstrate. 
The Replica dataset coordinate system show as the Fig.~\ref{fig:approach:coord_hotel_00}.

\begin{figure}[hbt!]
	\centering
	\includegraphics[width=\linewidth]{images/synthetic_optical_flow/coord_hotel_00.png}
	\caption{A boat.}
	\label{fig:approach:coord_hotel_00}
\end{figure}

And the coordinate system used in geometry OpenGL shader shown as Fig.~\ref{fig:approach:geometry_cs}.
The forward is $+z$, up is $+y$ and left is $+x$.
Meanwhile, the $\theta$ and 

\begin{figure}[hbt!]
	\centering
	\includegraphics[width=\linewidth]{images/synthetic_optical_flow/coord_hotel_00.png}
	\caption{A boat.}
	\label{fig:approach:geometry_cs}
\end{figure}



\section{Dataset}\label{sec:exp:data}

It's very hard to measure the real word optical flow information, especially the equirectangular image. 
And currently do not have any public optical flow equirectangular image dataset.
So we evaluate our method estimated optical flow quantity in the synthetic equirectangular optical flow dataset.
Meanwhile to analysis the performance we use SSIM e.t.c metrics to evaluate the optical flow warped the equirectangular image sequence in both real world dataset and synthetic dataset.


\subsection{Synthetic Dataset}\label{sec:exp:data:syn}

The rasterization render method OpenGL used to synthesize the ground truth optical flow.
The equirectangular optical flow generation algorithm introduction in Section. \ref{sec:app:panoof}.
Few public indoor 3D datasets are hired to render the ground truth data.

Although, the 3D computer graphics software toolsets, e.g. Blender have functions to estimate the object motion information, e.g motion vector of Blender. 
The rendered object motion used for motion blur e.t.c VFX is not accurate enough. 
There are obviously artefacts at the boundary of moving objects, Figure. ~\ref{fig:exp:blendermv}. 

\begin{figure}[hbt!]
	\centering
	\includegraphics[width=\linewidth]{example-image-a}
	\caption{Blender Motion Vector}
	\label{fig:exp:blendermv}
\end{figure}

\textbf{Replica 360} ~\cite{replica19arxiv}



\textbf{Matterport3D} ~\cite{Matterport3D}


\textbf{Real World Dataset}

We test our method on the real world dataset with warped image sequence.


\subsection{Panoramic Optical Flow}


\textbf{Regular icosahedron}
a convex polyhedron with 20 faces, 30 edges and 12 vertices. 

Reference:
\href{https://en.wikipedia.org/wiki/Regular_icosahedron}{wiki}
\href{https://mathworld.wolfram.com/GnomonicProjection.html}{Gnomonic Projection}
\href{https://mathworld.wolfram.com/RegularIcosahedron.html}{Weisstein, Eric W}
\href{https://math.wikia.org/wiki/Icosahedron}{Weisstein, Eric W}

\href{https://en.wikipedia.org/wiki/Gnomonic_projection}{Weisstein, Eric W}
\href{https://www.imo.net/observations/methods/visual-observation/minor/gnomonic/}{Weisstein, Eric W}

// --- LIBRARY INCLUDE -- //

#pragma region inc_lib
	#include <stdio.h>
	#include <stdbool.h>
	#include <math.h>
	#include <sys/time.h>

	#include <omp.h>

	#include "functions.h"

	#include "SDL2/SDL.h"
	#include "SDL2/SDL_opengl.h"

	#include "text.h"

	#include "cuda_runtime.h"
	#include "kernel.cuh"

#pragma endregion inc_lib

// --- VARIABLES CONSTANTES -- //

#pragma region constants

	static float g_inertia = 0.5f;

	static float oldCamPos[] = { 0.0f, 0.0f, -45.0f };
	static float oldCamRot[] = { 0.0f, 0.0f, 0.0f };
	static float newCamPos[] = { 0.0f, 0.0f, -45.0f };
	static float newCamRot[] = { 0.0f, 0.0f, 0.0f };

	static bool g_showGrid = false;
	static bool g_showAxes = true;

#pragma endregion constants

int main( int argc, char **argv){
	#pragma region initializing
		omp_set_num_threads(NUM_THREADS_CPU);

		SDL_Event event;
		SDL_Window * window;
		SDL_DisplayMode current;
		
		int width = windows_width;//640;
		int height = windows_height;//480;

		bool done = false;

		float mouseOriginX = 0.0f;
		float mouseOriginY = 0.0f;

		float mouseMoveX = 0.0f;
		float mouseMoveY = 0.0f;

		float mouseDeltaX = 0.0f;
		float mouseDeltaY = 0.0f;

		struct timeval begin, end;

		float fps = 0.0;
		char sfps[40] = "FPS: ";

		if ( SDL_Init ( SDL_INIT_EVERYTHING ) < 0 ) {
			printf( "error: unable to init sdl\n" );
			return -1;
		}

		if ( SDL_GetDesktopDisplayMode( 0, &current ) ) {
			printf( "error: unable to get current display mode\n" );
			return -1;
		}

		window = SDL_CreateWindow( "Galaxeirb", SDL_WINDOWPOS_CENTERED, 
												SDL_WINDOWPOS_CENTERED, 
												width, height, 
												SDL_WINDOW_OPENGL );

		SDL_GLContext glWindow = SDL_GL_CreateContext( window );

		GLenum status = glewInit();

		if ( status != GLEW_OK ) {
			printf( "error: unable to init glew\n" );
			return -1;
		}

		if ( ! InitTextRes( "./bin/DroidSans.ttf" ) ) {
			printf( "error: unable to init text resources\n" );
			return -1;
		}

		SDL_GL_SetSwapInterval(1);

		// Opening file

		FILE *fd;

		if ((fd = fopen("dubinski.tab","r")) == NULL){
			fprintf(stderr,"Impossible d'ouvrir le fichier données en lecture\n");
			exit(1);
		}
		// Selection des lignes

		int intervall = 81920/Nb_de_pts;
		char test[200];

	#pragma endregion initializing

	float pox[Nb_de_pts];
	float poy[Nb_de_pts];
	float poz[Nb_de_pts];
	
	float vex[Nb_de_pts];
	float vey[Nb_de_pts];
	float vez[Nb_de_pts];
	
	float mas[Nb_de_pts];
	
	float gal[Nb_de_pts];
	

	int line_in_file=0;

	fscanf(fd,"%f %f %f %f %f %f %f \n",	&mas[0],
											&pox[0],
											&poy[0],
											&poz[0],
											&vex[0],
											&vey[0],
											&vez[0]);
	gal[0] = 1; // Milky Way
	for(int j=1;j<Nb_de_pts;j++){
		fscanf(fd,"%f %f %f %f %f %f %f \n",	&mas[j],
												&pox[j],
												&poy[j],
												&poz[j],
												&vex[j],
												&vey[j],
												&vez[j]);
		line_in_file++;
		gal[j] = ((line_in_file < 16384) || ((line_in_file > 32768) && (line_in_file < 40960)) || ((line_in_file > 49152) && (line_in_file < 65536))); //1 Milky way

        for(int i=0;i<intervall-1;i++) {
			fgets(test,200,fd);
			line_in_file++;
		}
    }

	#pragma region init_GPU

		cudaError_t cudaStatus;
		cudaStatus = cudaSetDevice(0);
		if(cudaStatus != cudaSuccess) printf( "error: unable to setup cuda device\n");
		else printf( "Cuda set up ok\n");

		// In pointer
		float *deviceIn_pox  = NULL;
		float *deviceIn_poy  = NULL;
		float *deviceIn_poz  = NULL;
		
		float *deviceIn_vex  = NULL;
		float *deviceIn_vey  = NULL;
		float *deviceIn_vez  = NULL;
		
		float *deviceIn_mas  = NULL;
		
		// Out pointer
		
		float *deviceOut_vex = NULL;
		float *deviceOut_vey = NULL;
		float *deviceOut_vez = NULL;
		
		// Memory allocation In

		CUDA_MALLOC((void**)&deviceIn_mas, sizeof(float)*Nb_de_pts);

		CUDA_MALLOC((void**)&deviceIn_pox, sizeof(float)*Nb_de_pts);
		CUDA_MALLOC((void**)&deviceIn_poy, sizeof(float)*Nb_de_pts);
		CUDA_MALLOC((void**)&deviceIn_poz, sizeof(float)*Nb_de_pts);
		
		// Memory allocation Out

		CUDA_MALLOC((void**)&deviceOut_vex, sizeof(float)*Nb_de_pts);
		CUDA_MALLOC((void**)&deviceOut_vey, sizeof(float)*Nb_de_pts);
		CUDA_MALLOC((void**)&deviceOut_vez, sizeof(float)*Nb_de_pts);
		
		
	#pragma endregion init_GPU

	while (!done){
		#pragma region init_nd_calc_SDL
			while ( SDL_PollEvent( &event ) ) {
				unsigned int e = event.type;
				if(SDL_MOUSEWHEEL){
					if(event.wheel.y == 1) // scroll up
					{
						oldCamPos[2] += (10/100.0f ) * 0.5 * fabs( oldCamPos[2]);
						oldCamPos[2] = oldCamPos[2] > -5.0f ? -5.0f : oldCamPos[2];
					}
					else if(event.wheel.y == -1) // scroll down
					{
						oldCamPos[2] -= (10/100.0f ) * 0.5 * fabs( oldCamPos[2]);
						oldCamPos[2] = oldCamPos[2] > -5.0f ? -5.0f : oldCamPos[2];
					}
				}
				if ( e == SDL_MOUSEMOTION ) {
					mouseMoveX = event.motion.x;
					mouseMoveY = height - event.motion.y - 1;
					
				} else if ( e == SDL_KEYDOWN ) {
					if ( event.key.keysym.sym == SDLK_F1 ) {
						g_showGrid = !g_showGrid;
					} else if ( event.key.keysym.sym == SDLK_F2 ) {
						g_showAxes = !g_showAxes;
					} else if ( event.key.keysym.sym == SDLK_ESCAPE ) {
						done = true;
					} else if ( event.key.keysym.sym == SDLK_r ) { // Restart
						line_in_file=0;
						fclose(fd);
						fd = fopen("dubinski.tab","r");
						fscanf(fd,"%f %f %f %f %f %f %f \n",	&mas[0],
																&pox[0],
																&poy[0],
																&poz[0],
																&vex[0],
																&vey[0],
																&vez[0]);
						gal[0] = 1; // Milky Way
						for(int j=1;j<Nb_de_pts;j++){
							fscanf(fd,"%f %f %f %f %f %f %f \n",	&mas[j],
																	&pox[j],
																	&poy[j],
																	&poz[j],
																	&vex[j],
																	&vey[j],
																	&vez[j]);
							line_in_file++;
							gal[j] = ((line_in_file < 16384) || ((line_in_file > 32768) && (line_in_file < 40960)) || ((line_in_file > 49152) && (line_in_file < 65536))); //1 Milky way

							for(int i=0;i<intervall-1;i++) {
								fgets(test,200,fd);
								line_in_file++;
							}
						}
					}
				}

				if ( e == SDL_QUIT ) {
					printf( "quit\n" );
					done = true;
				}
			}

			mouseDeltaX = mouseMoveX - mouseOriginX;
			mouseDeltaY = mouseMoveY - mouseOriginY;

			if ( SDL_GetMouseState( 0, 0 ) & SDL_BUTTON_LMASK ) {
				oldCamRot[ 0 ] += -mouseDeltaY / 5.0f;
				oldCamRot[ 1 ] += mouseDeltaX / 5.0f;
			}else if ( SDL_GetMouseState( 0, 0 ) & SDL_BUTTON_RMASK ) {
				oldCamPos[ 2 ] += ( mouseDeltaY / 100.0f ) * 0.5 * fabs( oldCamPos[ 2 ] );
				oldCamPos[ 2 ]  = oldCamPos[ 2 ] > -5.0f ? -5.0f : oldCamPos[ 2 ];
			}
			if(1){ // Initialisation
				mouseOriginX = mouseMoveX;
				mouseOriginY = mouseMoveY;

				glViewport( 0, 0, width, height );
				glClearColor( 0.2f, 0.2f, 0.2f, 1.0f );
				glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
				glEnable( GL_BLEND );
				glBlendEquation( GL_FUNC_ADD );
				glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
				glDisable( GL_TEXTURE_2D );
				glEnable( GL_DEPTH_TEST );
				glMatrixMode( GL_PROJECTION );
				glLoadIdentity();
				gluPerspective( 50.0f, (float)width / (float)height, 0.1f, 100000.0f );
				glMatrixMode( GL_MODELVIEW );
				glLoadIdentity();

				for (int w=0;w<3;++w) {
					newCamPos[w] += (oldCamPos[w] - newCamPos[w] ) * g_inertia;
					newCamRot[w] += (oldCamRot[w] - newCamRot[w] ) * g_inertia;
				}

				glTranslatef( newCamPos[0], newCamPos[1], newCamPos[2] );
				glRotatef(newCamRot[0], 1.0f, 0.0f, 0.0f );
				glRotatef(newCamRot[1], 0.0f, 1.0f, 0.0f );
				
				if ( g_showGrid ) {
					DrawGridXZ( -100.0f, 0.0f, -100.0f, 20, 20, 10.0 );
				}

				if ( g_showAxes ) {
					ShowAxes();
				}

				gettimeofday(&begin, NULL );
			}

		#pragma endregion init_nd_calc_SDL

		// Simulation should be computed here


		// Memory feed In

		for (int o=0;o<10;o++) printf("bf vex(%d):%.4f\n",o,vex[o]);

		 
		CUDA_MEMCPY(deviceIn_pox, pox, sizeof(float)*Nb_de_pts, cudaMemcpyHostToDevice); // Feed GPU with data
		CUDA_MEMCPY(deviceIn_poy, poy, sizeof(float)*Nb_de_pts, cudaMemcpyHostToDevice); // Feed GPU with data
		CUDA_MEMCPY(deviceIn_poz, poz, sizeof(float)*Nb_de_pts, cudaMemcpyHostToDevice); // Feed GPU with data
		
		CUDA_MEMCPY(deviceIn_mas, mas, sizeof(float)*Nb_de_pts, cudaMemcpyHostToDevice); // Feed GPU with data
		
		acc_calc(NUM_BLOCKS_GPU, NUM_THREADS_GPU, deviceIn_mas, deviceIn_pox, deviceIn_poy, deviceIn_poz, deviceOut_vex, deviceOut_vey, deviceOut_vez); // Make calc
		
		cudaStatus = cudaDeviceSynchronize(); // Wait for the end of calc
		if(cudaStatus != cudaSuccess) printf("error: unable to synchronize threads\n");

		CUDA_MEMCPY(vex, deviceOut_vex, sizeof(float)*Nb_de_pts, cudaMemcpyDeviceToHost); // Feed GPU with data
		CUDA_MEMCPY(vey, deviceOut_vey, sizeof(float)*Nb_de_pts, cudaMemcpyDeviceToHost); // Feed GPU with data
		CUDA_MEMCPY(vez, deviceOut_vez, sizeof(float)*Nb_de_pts, cudaMemcpyDeviceToHost); // Feed GPU with data
		
		/*
		for(int i =0;i < Nb_de_pts;i++){
			float dist;
			float tmp;

			float tx = 0.0;
			float ty = 0.0;
			float tz = 0.0;

			for(int j=0;j<Nb_de_pts;j++){

				if(j != i){
					dist = sqrtf(_Square(pox[j] - pox[i])+_Square(poy[j] - poy[i])+_Square(poz[j] - poz[i]));
					if(dist < 1.0) dist = 1.0;
					tmp = mass_factor_X_damping_factor * (1/_Cube(dist)) * mas[i];
					tx += ((pox[j] - pox[i]) * tmp);
					ty += ((poy[j] - poy[i]) * tmp);
					tz += ((poz[j] - poz[i]) * tmp);
				}
			}
			vex[i] = tx;
			vey[i] = ty;
			vez[i] = tz;
		}
		*/
		for (int o=0;o<10;o++) printf("af vex(%d):%.4f\n",o,vex[o]);
		
		for (int k=0;k<Nb_de_pts;k++){ // 2nd for each particle
			pox[k] += (vex[k] * time_factor);
			poy[k] += (vey[k] * time_factor);
			poz[k] += (vez[k] * time_factor);
		}

		glPointSize(1.0);
		glBegin(GL_POINTS);
		
		for (int k=0;k<Nb_de_pts;k++){
			glColor3f(gal[k], 1-gal[k] , 0);
			glVertex3f(pox[k], poy[k], poz[k]);
		}
		glEnd();
		
		
		#pragma region FPS_nd_draw

			gettimeofday( &end, NULL );

			fps = (float)1.0f / ( ( end.tv_sec - begin.tv_sec ) * 1000000.0f + end.tv_usec - begin.tv_usec) * 1000000.0f;
			sprintf( sfps, "FPS : %.4f", fps );
			printf("FPS : %.1f\n", fps);

			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			gluOrtho2D(0, width, 0, height);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			DrawText( 10, height - 20, sfps, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );

			char pts_c[20];
			sprintf(pts_c, "Nb de pts : %d", Nb_de_pts);
			DrawText( 10, height - 40, pts_c, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );

			char tfac_c[22];
			sprintf(tfac_c, "Time factor : %.4f", time_factor);
			DrawText( 10, height - 60, tfac_c, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );


			DrawText( 10, 30, "'F1' : show/hide grid", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
			DrawText( 10, 10, "'F2' : show/hide axes", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );

			SDL_GL_SwapWindow( window );
			SDL_UpdateWindowSurface( window );
		#pragma endregion FPS_nd_draw
	}

	#pragma region closing_SDL
		SDL_GL_DeleteContext( glWindow );
		DestroyTextRes();
		SDL_DestroyWindow( window );
		SDL_Quit();

		fclose(fd);
	#pragma endregion closing_SDL

	return 1;
}


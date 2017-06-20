// written by André Betz 
// http://www.andrebetz.de

/* Backpropagation Feedforward Net */
#include <stdio.h>	/* printf(), srand(), rand(), NULL */
#include <stdlib.h>	/* malloc(), free() */
#include <time.h>	/* time() */
#include <math.h>	/* exp() (use: gcc backprop.xx -lm) */

class BackPropNet
{
private:
	int	   m_Lay;
	int	  *m_Neur;
	float	   m_Gam;
	float	  *m_Gain;
	float	 **m_Out;
	float	 **m_Delt;
	float	***m_Weig;
public:
	BackPropNet	(int Lay,int *Neu,float gam);
	~BackPropNet	();
	void	Init	();
	void	Input	(float *In, float *Gain);
	void	Output	(float *Out);
	void	Calc	();
	void	Learn	();
};

BackPropNet::BackPropNet(int Lay,int *Neu,float gam)
{
	int x,y;
	m_Lay  = Lay;
	m_Gam  = gam;	
        
	m_Neur = (int*)malloc(sizeof(int) * m_Lay);
	for(x=0;x<m_Lay;x++)        
	{
		m_Neur[x] = Neu[x];
	}

	m_Gain = (float*)  malloc(sizeof(float)   * m_Neur[m_Lay - 1]);
	m_Out  = (float**) malloc(sizeof(float*)  * m_Lay);
	
	for(x=0;x<m_Lay;x++)
	{
		m_Out[x] = (float*)malloc(sizeof(float) * m_Neur[x]);
	}

	m_Weig = (float***)malloc(sizeof(float**) * m_Lay);
	m_Delt = (float**) malloc(sizeof(float**) * m_Lay);
	m_Weig[0] = NULL;
	m_Delt[0] = NULL;	

	for(x=1;x<m_Lay;x++)
	{
		m_Weig[x] = (float**)malloc(sizeof(float*) * m_Neur[x]);
		m_Delt[x] = (float*) malloc(sizeof(float*) * m_Neur[x]);
		
		for(y=0;y<m_Neur[x];y++)
		{
			m_Weig[x][y] = (float*)malloc(sizeof(float) * (m_Neur[x-1] + 1));
		}
	}

	Init();
}

BackPropNet::~BackPropNet()
{
	int x,y;
	
	for(x=1;x<m_Lay;x++)
	{
		for(y=0;y<m_Neur[x];y++)
		{
			free(m_Weig[x][y]);
		}
		free(m_Delt[x]);
		free(m_Weig[x]);
	}
	free(m_Delt);
	free(m_Weig);
	for(x=0;x<m_Lay;x++)
	{
		free(m_Out[x]);
	}
	free(m_Out);
	free(m_Gain);
}

void BackPropNet::Init()
{
	int x,y,z;
	float h;

	srand(time(NULL));

	for(x=1;x<m_Lay;x++)
	{
		for(y=0;y<m_Neur[x];y++)
		{
			for(z=0;z<=m_Neur[x-1];z++)
			{
				/* Random weights */
				h = (rand() % 100) / 100.0;
				m_Weig[x][y][z] = h;
			}
		}
	}
}

void BackPropNet::Input(float *In,float *Gain)
{
	int x;
	
	for(x=0;x<m_Neur[0];x++)
	{
		m_Out[0][x] = In[x];
	}

	for(x=0;x<m_Neur[m_Lay-1];x++)
	{
		m_Gain[x] = Gain[x];
	}
}

void BackPropNet::Output(float *Out)
{
	int x;
	
	for(x=0;x<m_Neur[m_Lay-1];x++)
	{
		Out[x] = m_Out[m_Lay-1][x];
	}
}

void BackPropNet::Calc()
{
	int x,y,z;
	float h;
	for(x=1;x<m_Lay;x++)
	{
		for(y=0;y<m_Neur[x];y++)
		{
			h = -0.0f;
			for(z=0;z<m_Neur[x-1];z++)
                        {
                        	h += m_Out[x-1][z] * m_Weig[x][y][z];
                        }
 
        		h += m_Weig[x][y][z];
			m_Out[x][y] = 1.0 / (1.0 + exp(-h));
		}
	}
}

void BackPropNet::Learn()
{
	int x,y,z;
	float h;

	Calc();
	/* Backpropagation Delta */
    	for(x=0;x<m_Neur[m_Lay-1];x++)
    	{  
       		h = m_Gain[x] - m_Out[m_Lay-1][x]; /* Delta */
		/* abgeleitete Sigmoid funktion */
		h *= m_Out[m_Lay-1][x] * (1.0 - m_Out[m_Lay-1][x]); 
        	m_Delt[m_Lay-1][x] = h;
    	}

	for(x=m_Lay-1;x>1;x--)
	{
		for(y=0;y<m_Neur[x-1];y++)
		{
			h = -0.0f;
			for(z=0;z<m_Neur[x];z++)
			{
				h += m_Delt[x][z] * m_Weig[x][z][y];
			}
			m_Delt[x-1][y] = m_Out[x-1][y] * (1.0 - m_Out[x-1][y]) * h;
		}
	}
	
	/* errorcorrection */
	for(x=1;x<m_Lay;x++)
	{
		for(y=0;y<m_Neur[x];y++)
		{
			for(z=0;z<m_Neur[x-1];z++)
			{
               			h = m_Out[x-1][z] * m_Gam;
				m_Weig[x][y][z] += m_Delt[x][y] * h; 
			}
			m_Weig[x][y][z] += m_Gam * m_Delt[x][y];
		}
	}	
}

void main()
{	
	float x1[] = {0.0,0.0}, y1[] = {0.0};
	float x2[] = {0.0,1.0}, y2[] = {1.0};
    	float x3[] = {1.0,0.0}, y3[] = {1.0};
    	float x4[] = {1.0,1.0}, y4[] = {0.0};
	float y[1];	
	int neuron[] = {2,2,1}, lay = 3,c=0;
    	int p = 5000;
   	
	BackPropNet netz(lay,neuron,0.25);

	for(;;)
	{	
		c++;
		if(p<c) printf("\n");
		netz.Input(x1,y1);
		netz.Learn();
		netz.Calc();
		netz.Output(y);
		if(p<c) printf("0 xor 0 = %f \n",y[0]);

        	netz.Input(x2,y2);
        	netz.Learn();
        	netz.Calc();
        	netz.Output(y);
        	if(p<c) printf("0 xor 1 = %f \n",y[0]);

        	netz.Input(x3,y3);
        	netz.Learn();
        	netz.Calc();
		netz.Output(y);
        	if(p<c) printf("1 xor 0 = %f \n",y[0]);

        	netz.Input(x4,y4);
        	netz.Learn();
        	netz.Calc();
        	netz.Output(y);
        	if(p<c) printf("1 xor 1 = %f \n\n",y[0]);

		if(p<c) getchar();
	}
}

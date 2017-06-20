// written by André Betz 
// http://www.andrebetz.de

/* Perceptron */
#include<stdio.h>	/* printf(), srand(), rand(), NULL */
#include<stdlib.h>	/* malloc(), free() */
#include<time.h>	/* time() */

class Perceptron
{
public:
	int 	*m_In;
	float	*m_Mat;
	float	 m_Gam;
	int	 m_Bias;
	int	 m_Out;
	int	 m_Gain;
	int	 m_long;
public:
	Perceptron(int In,int m_Bias,float Gam);
	~Perceptron();
	void Init();
	int Learn();
	void Calc();
	void Input(int* In,int Gain);
	int Output();	
};

Perceptron::Perceptron(int In,int Bias,float Gam)
{
	m_long = In + 1;
	m_Gam  = Gam;
	m_Bias = -Bias;
	m_Mat = (float*)malloc(sizeof(float) * m_long);
	m_In = (int*)  malloc(sizeof(int)   * m_long);

	Init();
}

Perceptron::~Perceptron()
{
	free(m_In);
	free(m_Mat);
}

void Perceptron::Init()
{
	int i;
	float h;

	srand(time(NULL));
	
	for(i=0;i<m_long;i++)
	{
		m_Mat[i] = (rand() % 100) / 100.0; 
	}
}

void Perceptron::Calc()
{
	int i;
	float m = -0.0f;
	
	for(i=0;i<m_long;i++)
	{
		m += m_Mat[i] * (float)(m_In[i]);
	}	

	if(m > 0) 	m_Out = 1;
	else		m_Out = 0;
}

int Perceptron::Learn()
{
	int i,erg,res;
	
	Calc();
		
	res = m_Gain - m_Out;

	for(i=0;i<m_long;i++)
	{
		m_Mat[i] += m_In[i] * m_Gam * res; 
	}

	return res;
}

void Perceptron::Input(int* In,int Gain)
{
 	int i;
	
	m_Gain = Gain;
	for(i=0;i<(m_long-1);i++)
	{
		m_In[i] = In[i];
	}
	m_In[i] = m_Bias;
}

int Perceptron::Output()
{
	return m_Out;
}

void main()
{
	int y,x;

	/* AND Funktion */                                                
	int  x1[2] = {0,0}, y1 = 0;                                       
	int  x2[2] = {0,1}, y2 = 0;                                           
	int  x3[2] = {1,0}, y3 = 0;                                          
	int  x4[2] = {1,1}, y4 = 1;   
	
	y = 1;
	x = 0;

	Perceptron net(2,1,0.5f);

	while(y)
	{
		y = 0;
		x++;
	
		net.Input(x1,y1);
		if(net.Learn()) y = 1;

		net.Input(x2,y2);
		if(net.Learn()) y = 1;

		net.Input(x3,y3);
		if(net.Learn()) y = 1;

		net.Input(x4,y4);
		if(net.Learn()) y = 1;

		net.Output();
	}
	printf("Iterations %d\n",x);
}

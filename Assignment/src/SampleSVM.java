import java.util.*;

public class SampleSVM {
	double lr;
	double lambda;
	long epochs;
	double[] weights;
	double bias;
	
	SampleSVM(double lr, double lambda, long epochs, int n_feats){
		this.lr = lr;
		this.lambda = lambda;
		this.epochs = epochs;
		this.weights = new double[n_feats];
		this.bias = 0;
	}
	
	public List<Double> getGradients(double[] x,int y){
		List<Double> output = new ArrayList<>();
		if(y*(dot(x,this.weights)+this.bias)>=1) {
			output = dot(this.weights,this.lambda);
			output.add((double)0);
			return output;
		}
		output = dot(this.weights,this.lambda,x,y);
		output.add((double)-y);
		return output;
	}
	
	private List<Double> dot(double[] arr,double c) {
		List<Double> ans = new ArrayList<>(arr.length);
		for(int i=0;i<arr.length;i++) {
			ans.add(arr[i]*c);
		}
		return ans;
	}
	
	private List<Double> dot(double[] arr,double c, double[] x, double y) {
		List<Double> ans = new ArrayList<>(arr.length);
		for(int i=0;i<arr.length;i++) {
			ans.add(arr[i]*c-x[i]*y);
		}
		return ans;
	}
	
	private double dot(double[] arr1,double[] arr2) {
		double sum=0;
		for(int i=0;i<arr1.length;i++) {
			sum+=arr1[i]*arr2[i];
		}
		return sum;
	}
}

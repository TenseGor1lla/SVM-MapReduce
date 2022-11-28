import java.io.*;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class SVM {
	
	// 
	public static List<Double> getGradients(double[] x,int y, double[] weights, double bias, double lambda){
		List<Double> output = new ArrayList<>();
		if(y*(dot(x,weights)+bias)>=1) {
			output = dot(weights,lambda);
			output.add((double)0);
			return output;
		}
		output = dot(weights,lambda,x,y);
		output.add((double)-y);
		return output;
	}
	
	// returns scalar multiplication of constant and a vector
	private static List<Double> dot(double[] arr,double c) {
		List<Double> ans = new ArrayList<>(arr.length);
		for(int i=0;i<arr.length;i++) {
			ans.add(arr[i]*c);
		}
		return ans;
	}
	
	// 
	private static List<Double> dot(double[] arr,double c, double[] x, double y) {
		List<Double> ans = new ArrayList<>(arr.length);
		for(int i=0;i<arr.length;i++) {
			ans.add(arr[i]*c-x[i]*y);
		}
		return ans;
	}
	
	// returns scalar product of two same sized vectors 
	private static double dot(double[] arr1,double[] arr2) {
		double sum=0;
		for(int i=0;i<arr1.length;i++) {
			sum+=arr1[i]*arr2[i];
		}
		return sum;
	}
	
	// 
	public static void applyGradients(double[] dw, double db, double[] weights, double bias, double lr) {
		List<Double> w = dot(dw,lr);
		for(int i=0;i<w.size();i++) {
			weights[i] -= w.get(i)/22500000;
		}
		bias-=db*lr/22500000;
	}
	
	public static class TextArrayWritable extends ArrayWritable {
        public TextArrayWritable() {
            super(Text.class);
        }

        public TextArrayWritable(String[] strings) {
            super(Text.class);
            Text[] texts = new Text[strings.length];
            for (int i = 0; i < strings.length; i++) {
                texts[i] = new Text(strings[i]);
            }
            set(texts);
        }
    }
	
	public static class TokenizerMapper extends Mapper<Object, Text, Text, TextArrayWritable>{
		private Text grads;
		TextArrayWritable arrayWritable;
		double lr;
		double lambda;
		double[] weights;
		double bias;
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			grads = new Text("Gradients");
			arrayWritable=new TextArrayWritable();
			lr = Double.parseDouble(context.getConfiguration().get("lr"));
			lambda = Double.parseDouble(context.getConfiguration().get("lambda"));
			String[] w = context.getConfiguration().get("weights").split(" ");
			weights=new double[2];
			weights[0] = Double.parseDouble(w[0]); // null ptr error is coming here
			weights[1] = Double.parseDouble(w[1]);
			bias = Double.parseDouble(context.getConfiguration().get("bias"));
		}
		
	    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
		      StringTokenizer itr = new StringTokenizer(value.toString());
		      List<Double> output = new ArrayList<>();
		      while (itr.hasMoreTokens()) {
			        output.add(Double.parseDouble(itr.nextToken()));
		      }
		      double[] x = new double[output.size()-1];
		      for(int i=0;i<output.size()-1;i++) {
		    	  x[i] = output.get(i);
		      }
		      int y=(int)(double)output.get(output.size()-1);
		      if(y<=0) y=-1;
		      else y=1;
		      List<Double> temp = getGradients(x,y,weights,bias,lambda);
		      Text[] gradients = new Text[temp.size()];
		      for(int i=0;i<temp.size();i++){
		    	  gradients[i]= new Text(temp.get(i).toString());
		      }
		      arrayWritable.set(gradients);
		      context.write(grads,arrayWritable);
	    }
	}

	public static class IntSumCombiner extends Reducer<Text,TextArrayWritable,Text,TextArrayWritable> {
	    // use classifier mapping
		private Text grads;
		TextArrayWritable arrayWritable;
		double lr;
		double lambda;
		double[] weights;
		double bias;
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			grads = new Text("Gradients");
			arrayWritable=new TextArrayWritable();
			lr = Double.parseDouble(context.getConfiguration().get("lr"));
			lambda = Double.parseDouble(context.getConfiguration().get("lambda"));
			String[] w = context.getConfiguration().get("weights").split(" ");
			weights=new double[2];
			weights[0] = Double.parseDouble(w[0]);
			weights[1] = Double.parseDouble(w[1]);
			bias = Double.parseDouble(context.getConfiguration().get("bias"));
		}
	    public void reduce(Text key, Iterable<TextArrayWritable> values, Context context) throws IOException, InterruptedException {
	    	double[] sum = new double[weights.length+1];
	    	for (TextArrayWritable val: values) {
	    		int i=0;
	    		for (Writable writable: val.get()) {                 
	    			Text text = (Text)writable;  
	    			double value = Double.parseDouble(text.toString());                    // get
	    			sum[i++]+=value;
	    		}
	    	}
	    	Text[] gradients = new Text[sum.length];
		    for(int i=0;i<sum.length;i++){
		    	gradients[i]= new Text(sum[i]+"");
		    }
		    arrayWritable.set(gradients);
		    context.write(grads,arrayWritable);
	    }
	}

	
	
	public static class IntSumReducer extends Reducer<Text,TextArrayWritable,Text,TextArrayWritable> {
	    // use classifier mapping
		
		ArrayWritable arrayWritable;
		double lr;
		double lambda;
		double[] weights;
		double bias;
		FileSystem fs;
		int itr;
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			arrayWritable = new TextArrayWritable();
			lr = Double.parseDouble(context.getConfiguration().get("lr"));
			lambda = Double.parseDouble(context.getConfiguration().get("lambda"));
			String[] w = context.getConfiguration().get("weights").split(" ");
			weights=new double[2];
			weights[0] = Double.parseDouble(w[0]);
			weights[1] = Double.parseDouble(w[1]);
			bias = Double.parseDouble(context.getConfiguration().get("bias"));
			fs=FileSystem.get(context.getConfiguration());
			itr=Integer.parseInt(context.getConfiguration().get("itr"));
		}
	    public void reduce(Text key, Iterable<TextArrayWritable> values, Context context) throws IOException, InterruptedException {
	    	double[] sum = new double[weights.length+1];
	    	for (TextArrayWritable val: values) {
	    		int i=0;
	    		for (Writable writable: val.get()) {                 
	    			Text text = (Text)writable;  
	    			double value = Double.parseDouble(text.toString());                    // get
	    			sum[i++]+=value;
	    		}
	    	}
	    	double[] dw = new double[weights.length];
	    	for(int i=0;i<weights.length;i++){
	    		dw[i]=sum[i];
	    	}
	    	double db = sum[weights.length];
	    	applyGradients(dw,db,weights,bias,lr);
	    }
	    protected void cleanup(Context context) throws IOException,InterruptedException {
	    	FSDataOutputStream out = fs.create(new Path("hdfs:/BDAAssignment/Outputs/"+itr+"/weights"), true);
	    	BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(out));
		    for(double it: weights) {
		    	bw.write(it+"\n");
		    }
		    bw.close();
		    out = fs.create(new Path("hdfs:/BDAAssignment/Outputs/"+itr+"/bias"), true);
	    	bw = new BufferedWriter(new OutputStreamWriter(out));
		    bw.write(bias+"\n");
		    bw.close();
		    out = fs.create(new Path("hdfs:/BDAAssignment/weights.txt"), true);
	    	bw = new BufferedWriter(new OutputStreamWriter(out));
		    for(double it: weights) {
		    	bw.write(it+" ");
		    }
		    bw.write("\n");
		    bw.close();
		    out = fs.create(new Path("hdfs:/BDAAssignment/bias.txt"), true);
	    	bw = new BufferedWriter(new OutputStreamWriter(out));
		    bw.write(bias+"\n");
		    bw.close();
	    }
	}
	
	
	
	public static void main(String[] args) throws Exception {
		int epochs = 3;
		for(int i=0;i<epochs;i++) {
		    Configuration conf = new Configuration();
		    Path train=new Path("hdfs:/BDAAssignment/train.txt");
		    Path weightsPath=new Path("hdfs:/BDAAssignment/weights.txt");
		    Path biasPath=new Path("hdfs:/BDAAssignment/bias.txt");
			FileSystem fs=FileSystem.get(conf);
			BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(weightsPath)));
			conf.set("weights", br.readLine());
			br.close();
			br=new BufferedReader(new InputStreamReader(fs.open(biasPath)));
			conf.set("bias", br.readLine());
			br.close();
		    conf.set("lr", "0.0001");
		    conf.set("lambda", "0.002");
		    conf.set("itr", i+"");
		    Job job = Job.getInstance(conf, "word count");
		    job.setJarByClass(SVM.class);
		    job.setMapperClass(TokenizerMapper.class);
		    job.setCombinerClass(IntSumCombiner.class);
		    job.setReducerClass(IntSumReducer.class);
		    job.setOutputKeyClass(Text.class);
		    job.setOutputValueClass(TextArrayWritable.class);
		    FileInputFormat.addInputPath(job, train);
		    Path path = new Path("hdfs:/BDAAssignment/Outputs/"+i);
		    if (fs.exists(path)){
		    	fs.delete(path,true);
		    }
		    FileOutputFormat.setOutputPath(job, path);
		    job.waitForCompletion(true);
		}
	}
}

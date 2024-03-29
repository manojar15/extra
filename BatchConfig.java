package com.fisglobal.training;

import javax.sql.DataSource;

import org.springframework.batch.core.Job;
import org.springframework.batch.core.Step;
import org.springframework.batch.core.configuration.annotation.EnableBatchProcessing;
import org.springframework.batch.core.configuration.annotation.JobBuilderFactory;
import org.springframework.batch.core.configuration.annotation.StepBuilderFactory;
import org.springframework.batch.core.launch.support.RunIdIncrementer;
import org.springframework.batch.core.listener.ChunkListenerSupport;
import org.springframework.batch.item.database.BeanPropertyItemSqlParameterSourceProvider;
import org.springframework.batch.item.database.JdbcBatchItemWriter;
import org.springframework.batch.item.file.FlatFileItemReader;
import org.springframework.batch.item.file.mapping.BeanWrapperFieldSetMapper;
import org.springframework.batch.item.file.mapping.DefaultLineMapper;
import org.springframework.batch.item.file.transform.DelimitedLineTokenizer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@EnableBatchProcessing
public class BatchConfig {
	@Autowired
	private DataSource dataSource;
	
	
	@Autowired
	private JobBuilderFactory jobBuilderFactory;
	
	@Autowired
	private StepBuilderFactory stepBuilderFactory;
	
	@Bean
	public FlatFileItemReader<User> reader()
	{
		FlatFileItemReader<User> reader = new FlatFileItemReader<>();
		reader.setResource(new ClassPathResource("records.csv"));
		reader.setLineMapper(getLineMapper());
		reader.setLinesToSkip(1);
		return reader;
	}
	
	@Bean
	public LineMapper<User> getLineMapper()
	{
		DefaultLineMapper<User> lineMapper = new DefaultLineMapper<>();
		DelimitedLineTokenizer lineTokenizer = new DelimitedLineTokenizer();
		lineTokenizer.setNames(new String[] {"Emp ID", "Name Prefix", "First Name", "Last Name"});
		lineTokenizer.setIncludedFields(new int[] {0,1,2,4});
		BeanWrapperFieldSetMapper<User> fieldSetMapper = new BeanWrapperFieldSetMapper<>();
		fieldSetMapper.setTargetType(User.class);
		lineMapper.setLineTokenizer(lineTokenizer);
		lineMapper.setFieldSetMapper(fieldSetMapper);
		return lineMapper;
	}
	
	@Bean
	public UserItemProcessor processor()
	{
		return new UserItemProcessor();
	}
	
	@Bean
	public JdbcBatchItemWriter<User> writer()
	{
		JdbcBatchItemWriter<User> writer = new JdbcBatchItemWriter<>();
		writer.setItemSqlParametersSourceProvider(new BeanPropertyItemSqlParameterSourceProvider<User>());
		writer.setSql("insert into user(userId,namePrefix,firstName,lastName) values(:userId,:namePrefix,:firstName,:lastName)");
		writer.setDataSource(this.dataSource);
		return writer;
	}
	
	@Bean
	public Job importUserJob()
	{
	return this.jobBuilderFactory.get("USER-IMPORT-JOB")
			.incrementer(new RunIdIncrementer())
			.flow(step1())
			.end()
			.build();
	}
	
	@Bean
	public Step step1()
	{
		this.stepBuilderFactory.get("step1")
		.<User, User>chunk(10)
		.reader(reader())
		.processor(processor())
		.writer(writer())
		.build();
	}
}

